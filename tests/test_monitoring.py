"""Phase 5: drift simulations, Evidently parsing, end-to-end monitoring outputs."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.common.config import Settings
from src.common.mlflow_utils import ChampionInfo
from src.data.schema import BILL_AMT_COLS, FEATURES, PAY_AMT_COLS, PAY_STATUS_COLS, TARGET
from src.monitoring import drift
from src.monitoring.run import run_monitoring
from tests.conftest import lr_pipeline

THRESHOLD = 0.5


@pytest.fixture(scope="module")
def fitted(small_xy):
    X, y = small_xy
    return lr_pipeline().fit(X, y)


@pytest.fixture(scope="module")
def frames(small_xy, fitted):
    """(reference with predictions, raw current features, current labels)."""
    X, y = small_xy
    ref_X, cur_X = X.iloc[:2000], X.iloc[2000:]
    ref = drift.score_batch(
        ref_X, y.iloc[:2000], lambda f: fitted.predict_proba(f)[:, 1], THRESHOLD
    )
    return ref, cur_X, y.iloc[2000:]


def _predict(model):
    return lambda f: model.predict_proba(f)[:, 1]


# ------------------------------------------------------------- simulations
def test_covariate_drift_shifts_only_targeted_columns(small_xy) -> None:
    X, _ = small_xy
    shifted = drift.simulate_covariate_drift(X)
    assert (shifted["LIMIT_BAL"] < X["LIMIT_BAL"]).all()
    pay = X[PAY_STATUS_COLS].to_numpy()
    expected = np.where(pay >= 0, np.minimum(pay + 1, 9), pay)  # -2 / -1 stay unchanged
    np.testing.assert_array_equal(shifted[PAY_STATUS_COLS].to_numpy(), expected)
    assert shifted[PAY_STATUS_COLS].to_numpy().max() <= 9
    assert (shifted[BILL_AMT_COLS].abs() >= X[BILL_AMT_COLS].abs()).all().all()
    pd.testing.assert_frame_equal(shifted[PAY_AMT_COLS], X[PAY_AMT_COLS])
    assert X is not shifted  # input untouched


def test_concept_drift_flips_expected_share_deterministically(small_xy) -> None:
    _, y = small_xy
    a, b = drift.simulate_concept_drift(y, seed=1), drift.simulate_concept_drift(y, seed=1)
    pd.testing.assert_series_equal(a, b)
    assert (a != y).mean() == pytest.approx(drift.CONCEPT_FLIP_RATE, abs=0.03)


# ------------------------------------------------------- evidently parsing
def test_report_parsing_and_metric_agreement(frames, fitted) -> None:
    ref, cur_X, cur_y = frames
    current = drift.score_batch(cur_X, cur_y, _predict(fitted), THRESHOLD)
    _, parsed = drift.run_report(ref, current, THRESHOLD)
    assert set(parsed["column_drift"]) == set(FEATURES)
    assert parsed["drift"]["drift_share"] == pytest.approx(
        parsed["drift"]["drifted_count"] / len(FEATURES)
    )
    assert {"f1", "accuracy", "precision", "recall", "roc_auc"} <= set(parsed["classification"])
    # Evidently's F1 must match sklearn's on the same thresholded predictions.
    assert parsed["classification"]["f1"] == pytest.approx(
        f1_score(current[TARGET], current[drift.PRED]), abs=1e-9
    )


def test_drift_detects_shift_but_not_same_distribution(frames, fitted) -> None:
    ref, cur_X, cur_y = frames
    baseline = drift.reference_quality(ref)
    same = drift.score_batch(cur_X, cur_y, _predict(fitted), THRESHOLD)
    shifted = drift.score_batch(
        drift.simulate_covariate_drift(cur_X), cur_y, _predict(fitted), THRESHOLD
    )
    s_same = drift.evaluate_scenario("same", "", ref, same, THRESHOLD, baseline)
    s_shift = drift.evaluate_scenario("shift", "", ref, shifted, THRESHOLD, baseline)
    assert s_same["alerts"]["data_drift"] is False
    assert s_shift["alerts"]["data_drift"] is True
    assert {"LIMIT_BAL", "PAY_1", "BILL_AMT1"} <= set(s_shift["drifted_columns"])
    assert not set(PAY_AMT_COLS) & set(s_shift["drifted_columns"])


# -------------------------------------------------------------- end to end
def test_run_monitoring_writes_reports_and_summary(small_xy, fitted, tmp_path, monkeypatch) -> None:
    X, y = small_xy
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}")
    settings = Settings(root=tmp_path)
    settings.ensure_dirs()

    # Hand-off files exactly as the training stage writes them.
    ref = X.iloc[:2000].astype("int64").copy()
    ref[TARGET] = y.iloc[:2000].to_numpy()
    ref[drift.PROBA] = fitted.predict_proba(X.iloc[:2000])[:, 1]
    ref[drift.PRED] = (ref[drift.PROBA] >= THRESHOLD).astype(int)
    ref.to_parquet(settings.holdout_dir / "train_reference.parquet", index=False)
    test = X.iloc[2000:].astype("int64").copy()
    test[TARGET] = y.iloc[2000:].to_numpy()
    test.to_parquet(settings.holdout_dir / "test.parquet", index=False)

    info = ChampionInfo(
        name="credit-default-pipeline",
        alias="champion",
        version="1",
        run_id="r1",
        threshold=THRESHOLD,
        input_columns=FEATURES,
        input_types=dict.fromkeys(FEATURES, "double"),
        tags={},
    )
    summary = run_monitoring(settings, model=fitted, info=info)

    for name in (
        "monitoring_report.html",
        "monitoring_report_covariate_drift.html",
        "monitoring_report_concept_drift.html",
    ):
        assert (settings.artifacts_dir / name).stat().st_size > 10_000
    saved = json.loads((settings.artifacts_dir / "monitoring_summary.json").read_text())
    assert saved["reference"]["n_rows"] == 2000 and saved["model"]["model_version"] == "1"
    by = {s["scenario"]: s for s in summary["scenarios"]}
    assert set(by) == {"holdout", "covariate_drift", "concept_drift"}
    assert by["covariate_drift"]["drift_share"] > by["holdout"]["drift_share"]
    assert by["concept_drift"]["drift_share"] == by["holdout"]["drift_share"]  # same features
    assert by["concept_drift"]["classification"]["f1"] < by["holdout"]["classification"]["f1"]
    assert by["concept_drift"]["alerts"]["f1_drop"] is True
    assert not any(by["holdout"]["alerts"].values())
    # Prediction-score drift: the downturn moves the score distribution, the holdout doesn't.
    assert by["covariate_drift"]["score_psi"] > by["holdout"]["score_psi"]
    assert by["holdout"]["score_psi"] < drift.SCORE_PSI_ALERT
    assert np.isfinite(by["holdout"]["classification"]["roc_auc"])


def test_parse_survives_missing_drift_summary() -> None:
    metrics = [
        {
            "config": {
                "type": "evidently:metric_v2:ValueDrift",
                "column": c,
                "method": "wasserstein",
                "threshold": 0.1,
            },
            "value": 0.5 if c == "LIMIT_BAL" else 0.01,
        }
        for c in FEATURES
    ]
    parsed = drift._parse({"metrics": metrics})  # no DriftedColumnsCount entry
    assert parsed["drift"]["drifted_count"] == 1
    assert parsed["drift"]["drift_share"] == pytest.approx(1 / len(FEATURES))
    assert parsed["drift"]["drifted_columns"] == ["LIMIT_BAL"]
