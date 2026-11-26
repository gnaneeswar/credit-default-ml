"""Phase 1: canonical schema, offline ingest, data dictionary, split, CV, EDA artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.config import Settings
from src.data import ingest, schema
from src.data.eda import run_eda
from src.data.split import make_cv, split_xy, train_test

REPO = Settings()


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return ingest.load_dataset(REPO)


def test_mapping_follows_uci_naming() -> None:
    m = schema.RAW_TO_CANONICAL
    assert m["X1"] == "LIMIT_BAL" and m["X6"] == "PAY_1" and m["X7"] == "PAY_2"
    assert m["X12"] == "BILL_AMT1" and m["X23"] == "PAY_AMT6" and m["Y"] == schema.TARGET
    # Standardised 1..6 indexing: UCI's PAY_0 is renamed to PAY_1 on load.
    assert schema.PAY_STATUS_COLS == ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    assert "PAY_0" not in schema.FEATURES and schema.UCI_ORIGINAL_NAMES == {"PAY_1": "PAY_0"}
    assert len(schema.FEATURES) == 23 and len(set(schema.FEATURES)) == 23


def test_cached_dataset_loads_offline(monkeypatch) -> None:
    def _no_network(*_a, **_k):
        raise AssertionError("network fetch attempted although cache exists")

    monkeypatch.setattr(ingest, "fetch_from_uci", _no_network)
    raw = ingest.load_raw(REPO)
    assert raw.shape == (30000, 25)


def test_canonical_frame(df: pd.DataFrame) -> None:
    assert list(df.columns) == [*schema.FEATURES, schema.TARGET]
    assert schema.ID_COL not in df.columns
    assert df.shape == (30000, 24)
    assert all(dt == np.int64 for dt in df.dtypes)
    assert set(df[schema.TARGET].unique()) == {0, 1}


def test_to_canonical_renames_and_orders() -> None:
    raw_cols = [schema.ID_COL, *reversed([f"X{i}" for i in range(1, 24)]), "Y"]
    raw = pd.DataFrame([range(len(raw_cols))], columns=raw_cols)
    out = ingest.to_canonical(raw)
    assert list(out.columns) == [*schema.FEATURES, schema.TARGET]
    assert out.loc[0, "LIMIT_BAL"] == raw.loc[0, "X1"]


def test_data_dictionary_is_complete_and_regenerable() -> None:
    def _reject_nan(token: str):
        raise ValueError(f"non-strict JSON constant {token}")

    d = json.loads(REPO.dictionary_json.read_text(encoding="utf-8"), parse_constant=_reject_nan)
    assert "\r" not in REPO.dictionary_md.read_bytes().decode("utf-8")
    assert d["dataset"]["uci_id"] == 350
    assert d["canonical_mapping"]["X6"] == "PAY_1"
    assert d["uci_original_names"] == {"PAY_1": "PAY_0"}
    assert [t["month"] for t in d["timeline"]][0] == "2005-09" and d["target_month"] == "2005-10"
    bm = d["billing_mechanics"]
    assert len(bm["delinquency_progression"]) == 4
    assert set(bm["accounting_rules"]) == {"BILL_AMT_n", "PAY_AMT_n", "PAY_n"}
    assert "domain_rule_checks" in bm["empirical_check"]
    assert {v["canonical_name"] for v in d["variables"]} >= set(schema.FEATURES)
    assert d["undocumented_codes"]["PAY_n"]["codes"] == [-2, 0]
    # The markdown is a pure rendering of the JSON, so it can be rebuilt offline.
    assert ingest.render_dictionary_md(d) == REPO.dictionary_md.read_text(encoding="utf-8")


def test_pay_status_definitions_in_dictionary() -> None:
    d = json.loads(REPO.dictionary_json.read_text(encoding="utf-8"))
    defs = {x["code"]: x["meaning"] for x in d["pay_status_definitions"]["definitions"]}
    assert defs == {
        -2: "Inactive cycle",
        -1: "Paid in full",
        0: "Revolving / minimum met",
        "1..9": "Delinquent",
    }
    md = REPO.dictionary_md.read_text(encoding="utf-8")
    assert "Revolving / minimum met" in md and "## Billing-cycle mechanics" in md


def test_dictionary_rebuilds_offline_idempotently(tmp_path: Path) -> None:
    s = Settings(root=tmp_path)
    s.dictionary_json.parent.mkdir(parents=True)
    s.dictionary_json.write_bytes(REPO.dictionary_json.read_bytes())
    ingest.rebuild_dictionary(s)
    assert s.dictionary_json.read_bytes() == REPO.dictionary_json.read_bytes()
    assert s.dictionary_md.read_bytes() == REPO.dictionary_md.read_bytes()


def test_stratified_split_is_deterministic(df: pd.DataFrame) -> None:
    train, test = train_test(df, REPO)
    assert (len(train), len(test)) == (24000, 6000)
    assert abs(train[schema.TARGET].mean() - test[schema.TARGET].mean()) < 1e-3
    train2, _ = train_test(df, REPO)
    pd.testing.assert_frame_equal(train, train2)


def test_cv_folds_cover_every_training_row_once(df: pd.DataFrame) -> None:
    train, _ = train_test(df, REPO)
    X, y = split_xy(train)
    cv = make_cv(REPO)
    assert cv.get_n_splits() == 3
    seen = np.concatenate([val for _, val in cv.split(X, y)])
    assert np.array_equal(np.sort(seen), np.arange(len(train)))


def test_eda_artifacts(df: pd.DataFrame, tmp_path: Path) -> None:
    summary = run_eda(df, Settings(root=tmp_path))
    assert (tmp_path / "artifacts" / "eda_summary.json").is_file()
    report = (tmp_path / "artifacts" / "eda_report.md").read_text(encoding="utf-8")
    assert "Undocumented codes" in report
    assert summary["total_missing"] == 0
    assert 0.21 < summary["target_balance"]["default_rate_full"] < 0.23
    u = summary["undocumented_codes"]
    assert set(u["EDUCATION"]) == {"0", "5", "6"} and set(u["MARRIAGE"]) == {"0"}
    assert all(set(u[c]) == {"-2", "0"} for c in schema.PAY_STATUS_COLS)
    cons = pd.DataFrame(summary["pay_code_bill_consistency"])
    assert len(cons) == 6 * 3
    revolving = cons[(cons["pay_col"] == "PAY_1") & (cons["code"] == 0)].iloc[0]
    assert revolving["share_bill_le_0"] < 0.01  # revolving => a balance exists

    checks = summary["domain_rule_checks"]
    assert {
        "status_code_counts",
        "timing_alignment",
        "inactive_vs_bill",
        "segments",
        "transition_counts",
    } <= set(checks)
    seg = checks["segments"]
    # Brief's risk ordering holds: never-late < cured < roll-forward < stagnant.
    assert seg["never_late"]["default_rate"] < seg["cured_now_current"]["default_rate"]
    assert seg["cured_now_current"]["default_rate"] < seg["roll_forward"]["default_rate"]
    assert seg["roll_forward"]["default_rate"] < seg["stagnant"]["default_rate"]
    assert seg["credit_balance"]["default_rate"] < seg["base"]["default_rate"]
    assert seg["over_limit"]["default_rate"] > seg["base"]["default_rate"]
    latest = checks["transition_counts"][-1]
    assert latest["transition"] == "PAY_2 -> PAY_1" and latest["dormant_to_revolver"] == 0
    assert "## Billing-cycle domain rules vs the data" in report
    # M5: label-dependent statistics use the training split only; counts use the full data.
    assert sum(r["count"] for r in summary["code_tables"]["EDUCATION"]) == 24000
    assert seg["base"]["n"] == 24000
    assert sum(summary["code_counts_full"]["EDUCATION"].values()) == 30000
    assert summary["label_dependent_stats_on"] == "train split (n=24000)"
