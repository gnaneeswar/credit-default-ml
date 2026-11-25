"""Automated EDA: terminal summaries + ``artifacts/eda_summary.json`` and ``eda_report.md``.

The EDA is descriptive and runs on the full dataset. It informs only *fixed, domain-defined*
rules (code remapping), never fitted parameters, so it introduces no train/test leakage.

CLI: ``python -m src.data.eda``
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.config import Settings, get_settings
from src.common.logging import get_logger, log_path_io, stage
from src.common.tables import log_table, render_table
from src.data import schema
from src.data.ingest import load_dataset
from src.data.split import train_test

log = get_logger(__name__)

_DOCUMENTED = {
    "SEX": set(schema.DOCUMENTED_CODES["SEX"]),
    "EDUCATION": set(schema.DOCUMENTED_CODES["EDUCATION"]),
    "MARRIAGE": set(schema.DOCUMENTED_CODES["MARRIAGE"]),
    **{c: set(schema.DOCUMENTED_CODES["PAY_n"]) for c in schema.PAY_STATUS_COLS},
}


def code_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-code count, share and default rate, flagging codes missing from the documentation."""
    g = df.groupby(col)[schema.TARGET].agg(count="size", default_rate="mean").reset_index()
    g["share"] = g["count"] / len(df)
    g["documented"] = g[col].isin(_DOCUMENTED[col])
    return g[[col, "count", "share", "default_rate", "documented"]]


def build_summary(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    target = df[schema.TARGET]
    codes = {c: code_table(df, c) for c in _DOCUMENTED}
    undocumented = {
        c: {
            str(int(r[c])): {"count": int(r["count"]), "default_rate": round(r["default_rate"], 4)}
            for _, r in t[~t["documented"]].iterrows()
        }
        for c, t in codes.items()
    }
    numeric = [
        "LIMIT_BAL",
        "AGE",
        *schema.BILL_AMT_COLS,
        *schema.PAY_AMT_COLS,
    ]
    return {
        "shapes": {
            "full": list(df.shape),
            "train": list(train.shape),
            "test": list(test.shape),
        },
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v},
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_balance": {
            "default_rate_full": round(float(target.mean()), 4),
            "default_rate_train": round(float(train[schema.TARGET].mean()), 4),
            "default_rate_test": round(float(test[schema.TARGET].mean()), 4),
            "counts": {str(k): int(v) for k, v in target.value_counts().sort_index().items()},
            "imbalance_ratio_neg_to_pos": round(float((target == 0).sum() / target.sum()), 3),
        },
        "numeric_distributions": json.loads(
            df[numeric].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(2).to_json()
        ),
        "negative_bill_rows": int((df[schema.BILL_AMT_COLS] < 0).any(axis=1).sum()),
        "zero_bill_all_months_rows": int((df[schema.BILL_AMT_COLS] == 0).all(axis=1).sum()),
        "code_tables": {
            c: json.loads(t.round(4).to_json(orient="records")) for c, t in codes.items()
        },
        "undocumented_codes": undocumented,
    }


def render_report(summary: dict, codes: dict[str, pd.DataFrame]) -> str:
    tb = summary["target_balance"]
    sh = summary["shapes"]
    pay0 = codes["PAY_0"]
    lines = [
        "# EDA Report — Default of Credit Card Clients",
        "",
        "## Shapes",
        "",
        f"- full: {sh['full'][0]} × {sh['full'][1]}  ·  train: {sh['train'][0]} × "
        f"{sh['train'][1]}  ·  test: {sh['test'][0]} × {sh['test'][1]}",
        f"- missing values: {summary['total_missing']}  ·  duplicate rows: "
        f"{summary['duplicate_rows']}",
        "- duplicate feature rows have distinct customer IDs, so they are kept as genuine records",
        f"- rows with any negative bill: {summary['negative_bill_rows']}  ·  rows with zero bill "
        f"in all 6 months: {summary['zero_bill_all_months_rows']}",
        "",
        "## Target balance",
        "",
        f"- default rate: full {tb['default_rate_full']:.2%}, train {tb['default_rate_train']:.2%},"
        f" test {tb['default_rate_test']:.2%} (stratified split preserves it)",
        f"- negative:positive ≈ {tb['imbalance_ratio_neg_to_pos']}:1 → accuracy is misleading; "
        "use F1 / PR-AUC and an explicit cutoff",
        "",
        "## Undocumented codes (evidence for handling decisions)",
        "",
    ]
    for col in ("EDUCATION", "MARRIAGE"):
        lines += [f"### {col}", "", render_table(codes[col]), ""]
    lines += [
        "→ EDUCATION 0/5/6 are rare (<1.2% combined) and their default rates sit close to",
        "  code 4 'others' and well below codes 1-3, so the data supports remapping them to 4.",
        "→ MARRIAGE 0 has too few rows to estimate its own risk reliably; it is merged into",
        "  3 'others' on semantic grounds (unknown ≈ not married/single), not because rates match.",
        "",
        "### PAY_0 (most recent month; PAY_2..PAY_6 behave the same — see eda_summary.json)",
        "",
        render_table(pay0),
        "",
        "→ `0` (revolving, minimum paid) is the most common code and `-2` (inactive) is common;",
        "  both default *less* often than `-1` (paid duly), so collapsing them into `-1` would",
        "  destroy signal. All PAY_n codes are kept as distinct ordinal values.",
        "",
        "## Numeric distributions",
        "",
        render_table(
            pd.DataFrame(summary["numeric_distributions"]).T.reset_index(names="feature"),
            floatfmt=".1f",
        ),
        "",
    ]
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    log_path_io(log, "write", path)


def run_eda(df: pd.DataFrame | None = None, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    df = load_dataset(settings) if df is None else df
    train, test = train_test(df, settings)
    summary = build_summary(df, train, test)
    codes = {c: code_table(df, c) for c in _DOCUMENTED}

    tb = summary["target_balance"]
    log.info(
        "missing=%s duplicates=%s default_rate=%.4f imbalance=%s:1",
        summary["total_missing"],
        summary["duplicate_rows"],
        tb["default_rate_full"],
        tb["imbalance_ratio_neg_to_pos"],
    )
    for col in ("EDUCATION", "MARRIAGE", "PAY_0"):
        log_table(log, f"code table: {col}", codes[col])
    log.info(
        "undocumented codes found: %s",
        {k: list(v) for k, v in summary["undocumented_codes"].items() if v},
    )

    _write(settings.artifacts_dir / "eda_summary.json", json.dumps(summary, indent=2) + "\n")
    _write(settings.artifacts_dir / "eda_report.md", render_report(summary, codes))
    return summary


def main() -> None:
    with stage("eda"):
        run_eda()


if __name__ == "__main__":
    main()
