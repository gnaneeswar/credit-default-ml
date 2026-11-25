"""Offline-first ingestion of UCI dataset 350 plus its data dictionary.

* The cached CSV under ``data/raw/`` (raw UCI columns: ID, X1..X23, Y) is committed to git,
  so cloning the repo is enough to train offline. ``ucimlrepo`` is only hit on a cache miss
  or with ``--refresh``.
* ``load_dataset()`` renames X1..X23/Y to canonical domain names and drops the ID.

CLI: ``python -m src.data.ingest [--refresh]``
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from src.common.config import Settings, get_settings
from src.common.logging import get_logger, log_path_io, stage
from src.common.tables import log_shape
from src.data import schema

log = get_logger(__name__)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert ucimlrepo dotdicts / pandas objects into plain JSON types.

    UCI free-text fields contain CRLF; strings are normalised to LF so generated files
    honour the repo's LF-only invariant.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, str):
        return obj.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(obj, float) and obj != obj:  # NaN -> null (strict JSON)
        return None
    if obj is None or isinstance(obj, int | float | bool):
        return obj
    if pd.isna(obj):
        return None
    return str(obj)


def fetch_from_uci(settings: Settings) -> pd.DataFrame:
    """Download dataset 350, write the raw CSV cache and the data dictionary."""
    from ucimlrepo import fetch_ucirepo  # network-only dependency, imported lazily

    log.info("fetching UCI dataset id=%s via ucimlrepo", schema.UCI_ID)
    ds = fetch_ucirepo(id=schema.UCI_ID)
    raw = ds.data.original
    _validate_raw(raw)

    settings.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(settings.raw_csv, index=False, lineterminator="\n")
    log_path_io(log, "write", settings.raw_csv)

    variables = ds.variables.where(pd.notna(ds.variables), None).to_dict(orient="records")
    write_dictionary(settings, build_dictionary(_to_jsonable(dict(ds.metadata)), variables))
    return raw


def build_dictionary(metadata: dict, variables: list[dict]) -> dict:
    canonical = {v["name"]: schema.RAW_TO_CANONICAL.get(v["name"], v["name"]) for v in variables}
    return {
        "dataset": {
            "uci_id": schema.UCI_ID,
            "name": metadata.get("name"),
            "repository_url": metadata.get("repository_url"),
            "doi": metadata.get("dataset_doi"),
            "num_instances": metadata.get("num_instances"),
            "creators": metadata.get("creators"),
            "intro_paper": metadata.get("intro_paper"),
        },
        "business_problem": schema.BUSINESS_PROBLEM,
        "target": {
            "raw": schema.RAW_TARGET,
            "canonical": schema.TARGET,
            "definition": "default payment next month (1 = default, 0 = no default)",
        },
        "variables": [
            {**_to_jsonable(v), "canonical_name": canonical[v["name"]]} for v in variables
        ],
        "canonical_mapping": schema.RAW_TO_CANONICAL,
        "timeline": schema.TIMELINE,
        "documented_codes": _to_jsonable(schema.DOCUMENTED_CODES),
        "undocumented_codes": schema.UNDOCUMENTED_CODES,
        "pay_status_definitions": {
            "source": schema.PAY_STATUS_DEFINITIONS_SOURCE,
            "definitions": schema.PAY_STATUS_DEFINITIONS,
        },
        "uci_variable_info": (metadata.get("additional_info") or {}).get("variable_info"),
        "source_metadata": metadata,
    }


def rebuild_dictionary(settings: Settings) -> dict:
    """Rebuild the dictionary offline from the UCI metadata/variables stored in its JSON."""
    old = json.loads(settings.dictionary_json.read_text(encoding="utf-8"))
    log_path_io(log, "read", settings.dictionary_json)
    variables = [
        {k: v for k, v in var.items() if k != "canonical_name"} for var in old["variables"]
    ]
    dictionary = build_dictionary(old["source_metadata"], variables)
    write_dictionary(settings, dictionary)
    return dictionary


def write_dictionary(settings: Settings, dictionary: dict) -> None:
    settings.dictionary_json.write_text(
        json.dumps(dictionary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    log_path_io(log, "write", settings.dictionary_json)
    settings.dictionary_md.write_text(
        render_dictionary_md(dictionary), encoding="utf-8", newline="\n"
    )
    log_path_io(log, "write", settings.dictionary_md)


def render_dictionary_md(d: dict) -> str:
    ds = d["dataset"]
    lines = [
        f"# Data Dictionary — {ds['name']} (UCI {ds['uci_id']})",
        "",
        f"Source: {ds['repository_url']} · DOI {ds['doi']} · {ds['num_instances']} rows",
        "",
        "## Business problem",
        "",
        d["business_problem"],
        "",
        "## Target",
        "",
        f"`{d['target']['raw']}` → `{d['target']['canonical']}`: {d['target']['definition']}",
        "",
        "## Variables",
        "",
        "| Raw | Canonical | Type | Role | Demographic |",
        "|---|---|---|---|---|",
    ]
    for v in d["variables"]:
        lines.append(
            f"| {v['name']} | `{v['canonical_name']}` | {v.get('type') or ''} "
            f"| {v.get('role') or ''} | {v.get('demographic') or ''} |"
        )
    lines += [
        "",
        "## Monthly timeline (position 1 = most recent)",
        "",
        "| Month | Repayment status | Bill amount | Amount paid |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {t['month']} | `{t['pay_status']}` | `{t['bill_amt']}` | `{t['pay_amt']}` |"
        for t in d["timeline"]
    ]
    lines += [
        "",
        "Note: `PAY_AMTn` is the amount paid *in* month n; it is conventionally read as settling "
        "the prior statement (`BILL_AMT(n+1)`), which is how payment-coverage ratios are built. "
        "There is no `PAY_1` in the original data.",
        "",
        "## Documented codes",
        "",
    ]
    for col, codes in d["documented_codes"].items():
        lines.append(f"- **{col}**: " + "; ".join(f"`{k}` = {v}" for k, v in codes.items()))
    lines += [
        "",
        "## Undocumented codes found in the data",
        "",
        "| Column | Codes | Interpretation | Pipeline handling |",
        "|---|---|---|---|",
    ]
    for col, u in d["undocumented_codes"].items():
        codes = ", ".join(f"`{c}`" for c in u["codes"])
        lines.append(f"| {col} | {codes} | {u['interpretation']} | {u['handling']} |")
    psd = d["pay_status_definitions"]
    lines += [
        "",
        f"## Low repayment-status codes — definitions ({psd['source']})",
        "",
        "| Code | Meaning | Financial behaviour |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{x['code']}` | {x['meaning']} | {x['financial_behavior']} |"
        for x in psd["definitions"]
    ]
    lines += [
        "",
        "Observed caveat: `-2` does not always coincide with a zero same-month bill (a balance,",
        "fees or a credit can still appear), so status codes and bill amounts are kept as",
        "separate signals — see `pay_code_bill_consistency` in `artifacts/eda_summary.json`.",
    ]
    lines += ["", "## UCI variable information (verbatim)", "", "```", d["uci_variable_info"]]
    lines += ["```", ""]
    return "\n".join(lines)


def _validate_raw(raw: pd.DataFrame) -> None:
    expected = {schema.ID_COL, *schema.RAW_TO_CANONICAL}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"raw dataset is missing columns: {sorted(missing)}")


def load_raw(settings: Settings | None = None, refresh: bool = False) -> pd.DataFrame:
    settings = settings or get_settings()
    if refresh or not settings.raw_csv.exists():
        return fetch_from_uci(settings)
    log_path_io(log, "read", settings.raw_csv)
    raw = pd.read_csv(settings.raw_csv)
    _validate_raw(raw)
    return raw


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename X1..X23/Y to domain names, drop the ID, and order columns canonically."""
    df = raw.rename(columns=schema.RAW_TO_CANONICAL).drop(columns=[schema.ID_COL])
    return df[[*schema.FEATURES, schema.TARGET]].astype("int64")


def load_dataset(settings: Settings | None = None, refresh: bool = False) -> pd.DataFrame:
    raw = load_raw(settings, refresh=refresh)
    log_shape(log, "raw (ID, X1..X23, Y)", raw)
    df = to_canonical(raw)
    log_shape(log, "canonical (23 features + target)", df)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download from UCI")
    parser.add_argument(
        "--rebuild-dictionary",
        action="store_true",
        help="regenerate data_dictionary.json/.md offline from the stored UCI metadata",
    )
    args = parser.parse_args(argv)
    with stage("ingest"):
        if args.rebuild_dictionary:
            rebuild_dictionary(get_settings())
        load_dataset(refresh=args.refresh)


if __name__ == "__main__":
    main()
