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


def pay_bill_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """For low PAY_n codes, how the same-month bill and payment actually look.

    Validates the domain definitions (-2 no consumption, -1 paid in full, 0 revolving)
    against the data rather than assuming them.
    """
    rows = []
    for p, b, a in zip(
        schema.PAY_STATUS_COLS, schema.BILL_AMT_COLS, schema.PAY_AMT_COLS, strict=True
    ):
        for code in (-2, -1, 0):
            sub = df[df[p] == code]
            rows.append(
                {
                    "pay_col": p,
                    "code": code,
                    "n": len(sub),
                    "share_bill_eq_0": (sub[b] == 0).mean(),
                    "share_bill_le_0": (sub[b] <= 0).mean(),
                    "share_pay_amt_eq_0": (sub[a] == 0).mean(),
                    "median_bill": float(sub[b].median()),
                    "default_rate": sub[schema.TARGET].mean(),
                }
            )
    return pd.DataFrame(rows)


SEGMENTS = {
    # name: (description, mask over engineered features `f` and raw frame `df`)
    "never_late": ("no month with PAY_n > 0", lambda f, df: f["N_DELINQ"] == 0),
    # Counts cover the history transitions and *_RECENT the latest one, so "any" = either.
    "cured_now_current": (
        "had a cure (PAY_(n+1) >= 2 -> PAY_n <= 0) and PAY_1 <= 0",
        lambda f, df: ((f["N_CURES"] > 0) | (f["CURED_RECENT"] > 0)) & (df["PAY_1"] <= 0),
    ),
    "roll_forward": (
        "any worsening into delinquency",
        lambda f, df: (f["N_ROLL_FWD"] > 0) | (f["ROLL_FWD_RECENT"] > 0),
    ),
    "roll_backward": (
        "any improvement while past due",
        lambda f, df: (f["N_ROLL_BACK"] > 0) | (f["ROLL_BACK_RECENT"] > 0),
    ),
    "stagnant": (
        "stuck at the same PAY >= 2",
        lambda f, df: (f["N_STAGNANT"] > 0) | (f["STAGNANT_RECENT"] > 0),
    ),
    "dormant_to_revolver": ("any -2 -> 0 activation", lambda f, df: f["N_DORMANT_TO_REVOLVER"] > 0),
    "over_limit": ("any BILL_AMT_n > LIMIT_BAL", lambda f, df: f["N_OVER_LIMIT"] > 0),
    "credit_balance": ("any BILL_AMT_n < 0", lambda f, df: f["N_CREDIT_BAL"] > 0),
    "paydown": ("any PAY_AMT_n > BILL_AMT_n", lambda f, df: f["N_PAYDOWN"] > 0),
}


def domain_rule_checks(df: pd.DataFrame, labelled: pd.DataFrame | None = None) -> dict:
    """Test the billing-cycle domain rules against the data.

    Label-free checks (status-code counts, timing alignment, inactive vs bill, transition
    counts) use ``df``. Label-dependent ones (segment default rates) use ``labelled`` — the
    training split in the pipeline — so holdout labels never inform design decisions.
    Segments are defined through DomainFeatureEngineer outputs, so the EDA and the model
    share a single definition of every rule.
    """
    from src.features.domain import DomainCleaner, DomainFeatureEngineer

    labelled = df if labelled is None else labelled
    f = DomainFeatureEngineer().fit_transform(
        DomainCleaner().fit_transform(labelled[schema.FEATURES])
    )
    y = labelled[schema.TARGET]

    timing = []
    for k, col in enumerate(schema.PAY_STATUS_COLS, start=1):
        paid = df[df[col] == -1]
        row: dict = {"status": col, "n_paid_in_full": len(paid)}
        if k < 6:  # brief's section-4 pairing: previous statement (k+1) settled in month k
            s = paid[paid[f"BILL_AMT{k + 1}"] > 0]
            row["rule_pairing"] = f"PAY_AMT{k} >= BILL_AMT{k + 1}"
            row["rule_share"] = float((s[f"PAY_AMT{k}"] >= s[f"BILL_AMT{k + 1}"]).mean())
        if k > 1:  # one month later: statement k settled in the following month (k-1)
            s = paid[paid[f"BILL_AMT{k}"] > 0]
            row["next_month_pairing"] = f"PAY_AMT{k - 1} >= BILL_AMT{k}"
            row["next_month_share"] = float((s[f"PAY_AMT{k - 1}"] >= s[f"BILL_AMT{k}"]).mean())
        timing.append(row)

    inactive = [
        {
            "status": p,
            "n_inactive": int((df[p] == -2).sum()),
            "share_bill_le_0": float((df.loc[df[p] == -2, b] <= 0).mean()),
        }
        for p, b in zip(schema.PAY_STATUS_COLS, schema.BILL_AMT_COLS, strict=True)
    ]

    segments = {
        "base": {"description": "all rows", "n": len(labelled), "default_rate": float(y.mean())}
    }
    for name, (desc, rule) in SEGMENTS.items():
        m = rule(f, labelled)
        segments[name] = {
            "description": desc,
            "n": int(m.sum()),
            "default_rate": float(y[m].mean()),
        }

    chrono = df[schema.PAY_STATUS_COLS[::-1]].to_numpy()
    transitions = []
    for j in range(chrono.shape[1] - 1):
        prev, cur = chrono[:, j], chrono[:, j + 1]
        transitions.append(
            {
                "transition": f"{schema.PAY_STATUS_COLS[::-1][j]} -> {schema.PAY_STATUS_COLS[::-1][j + 1]}",
                "cure": int(((prev >= 2) & (cur <= 0)).sum()),
                "roll_forward": int(((cur > prev) & (cur >= 1)).sum()),
                "roll_backward": int(((cur < prev) & (cur > 0) & (prev > 0)).sum()),
                "stagnant": int(((cur == prev) & (cur >= 2)).sum()),
                "dormant_to_revolver": int(((prev == -2) & (cur == 0)).sum()),
            }
        )
    codes = (
        pd.DataFrame({c: df[c].value_counts() for c in schema.PAY_STATUS_COLS})
        .reindex(range(-2, 10))
        .fillna(0)
        .astype(int)
    )
    return {
        "status_code_counts": {str(k): v for k, v in codes.to_dict(orient="index").items()},
        "timing_alignment": timing,
        "inactive_vs_bill": inactive,
        "segments": segments,
        "transition_counts": transitions,
    }


def build_summary(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Label-free statistics use the full data; anything computed from the target (default
    rates by code, segments, PAY consistency) uses the TRAINING split only."""
    target = df[schema.TARGET]
    codes = {c: code_table(train, c) for c in _DOCUMENTED}
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
        "label_dependent_stats_on": f"train split (n={len(train)})",
        "code_counts_full": {
            c: {str(int(k)): int(v) for k, v in df[c].value_counts().sort_index().items()}
            for c in _DOCUMENTED
        },
        "code_tables": {
            c: json.loads(t.round(4).to_json(orient="records")) for c, t in codes.items()
        },
        "undocumented_codes": undocumented,
        "pay_status_definitions": schema.PAY_STATUS_DEFINITIONS,
        "pay_code_bill_consistency": json.loads(
            pay_bill_consistency(train).round(4).to_json(orient="records")
        ),
        "domain_rule_checks": domain_rule_checks(df, labelled=train),
    }


def _pay1_consistency(summary: dict) -> pd.DataFrame:
    t = pd.DataFrame(summary["pay_code_bill_consistency"])
    return t[t["pay_col"] == "PAY_1"]


def _domain_rule_section(checks: dict) -> list[str]:
    seg = pd.DataFrame(checks["segments"]).T.reset_index(names="segment")
    seg["default_rate"] = seg["default_rate"].astype(float)
    return [
        "## Billing-cycle domain rules vs the data",
        "",
        "### Status-code counts per month",
        "",
        render_table(
            pd.DataFrame(checks["status_code_counts"])
            .T.loc[["-2", "-1", "0", "1", "2", "3"]]
            .reset_index(names="code")
        ),
        "",
        "→ Code `1` (one missed due date) is common in `PAY_1` but almost absent from",
        "  `PAY_2`..`PAY_6`: the source coded the latest month differently. Transition features",
        "  therefore isolate the latest transition in `*_RECENT` flags rather than relying only",
        "  on 6-month counts that mix both regimes.",
        "",
        "### Timing: which payment confirms a 'paid in full' status?",
        "",
        render_table(pd.DataFrame(checks["timing_alignment"]), floatfmt=".3f"),
        "",
        "→ The section-4 rule pairs status k with the previous statement (k+1) paid in month k.",
        "  The data matches a one-month-later pairing far better (statement k covered by the",
        "  next month's payment), so status codes are documented as the rule *plus* this check.",
        "",
        "### Is '-2 => BILL_AMT_n <= 0' true?",
        "",
        render_table(pd.DataFrame(checks["inactive_vs_bill"]), floatfmt=".3f"),
        "",
        "### Default rate by behavioural segment (definitions = engineered features)",
        "",
        render_table(seg, floatfmt=".4f"),
        "",
        "### Transition counts per month pair",
        "",
        render_table(pd.DataFrame(checks["transition_counts"])),
        "",
        "→ `-2 -> 0` (dormant -> revolver) never occurs in the latest transition (PAY_2 -> PAY_1),",
        "  so it is engineered only as a 6-month count, not as a most-recent-month flag.",
        "",
    ]


def render_report(summary: dict, codes: dict[str, pd.DataFrame]) -> str:
    tb = summary["target_balance"]
    sh = summary["shapes"]
    pay1 = codes["PAY_1"]
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
        f"Default rates below use the {summary['label_dependent_stats_on']} only: the 20%",
        "holdout's labels never inform design decisions. Row counts of the preprocessing",
        "steps use the full data (`code_counts_full`).",
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
        "### PAY_1 (most recent month; PAY_2..PAY_6 behave the same — see eda_summary.json)",
        "",
        render_table(pay1),
        "",
        "→ `0` (revolving, minimum paid) is the most common code and `-2` (inactive) is common;",
        "  both default *less* often than `-1` (paid duly), so collapsing them into `-1` would",
        "  destroy signal. All PAY_n codes are kept as distinct ordinal values.",
        "",
        "### Do the low codes match their definitions? (same-month bill / payment)",
        "",
        render_table(_pay1_consistency(summary)),
        "",
        "→ `0` (revolving) carries large balances and `-1` (paid in full) small ones, as defined.",
        "  `-2` (inactive cycle) often still shows a non-zero bill (residual balance, fees or",
        "  credits), so the pipeline never infers one signal from the other: `PAY_n == -2` counts",
        "  and `BILL_AMT <= 0` flags are separate engineered features.",
        "",
        *_domain_rule_section(summary["domain_rule_checks"]),
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
    codes = {c: code_table(train, c) for c in _DOCUMENTED}  # label-dependent: train only

    tb = summary["target_balance"]
    log.info(
        "missing=%s duplicates=%s default_rate=%.4f imbalance=%s:1",
        summary["total_missing"],
        summary["duplicate_rows"],
        tb["default_rate_full"],
        tb["imbalance_ratio_neg_to_pos"],
    )
    for col in ("EDUCATION", "MARRIAGE", "PAY_1"):
        log_table(log, f"code table: {col}", codes[col])
    log_table(
        log,
        "segment default rates (domain rules vs data)",
        pd.DataFrame(summary["domain_rule_checks"]["segments"]).T.reset_index(names="segment"),
    )
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
