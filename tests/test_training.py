"""Phase 3: tuning, decile threshold selection, skops allowlist, registry round-trip."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.common.config import Settings
from src.data.split import make_cv
from src.training.models import MODEL_SPECS, ModelSpec
from src.training.registry import trusted_types
from src.training.threshold import choose_threshold, decile_table
from src.training.tune import tune_model
from tests.conftest import REPO_ROOT, TINY_THRESHOLD, lr_pipeline


# ------------------------------------------------------------------ model specs
def test_model_grids_are_compact_and_single_threaded() -> None:
    assert set(MODEL_SPECS) == {"logistic_regression", "random_forest", "lightgbm", "xgboost"}
    for spec in MODEL_SPECS.values():
        assert 4 <= spec.n_candidates <= 8, spec.name
    rf = MODEL_SPECS["random_forest"].build(Settings(), 3.5).named_steps["model"]
    xgb = MODEL_SPECS["xgboost"].build(Settings(), 3.5).named_steps["model"]
    assert rf.n_jobs == 1 and xgb.n_jobs == 1
    assert xgb.tree_method == "hist" and xgb.scale_pos_weight == 3.5
    assert 100 <= rf.n_estimators <= 200 and 100 <= xgb.n_estimators <= 200


def test_tune_model_reports_all_cv_metrics(small_xy) -> None:
    X, y = small_xy
    spec = ModelSpec(
        "logistic_regression",
        "linear+woe",
        MODEL_SPECS["logistic_regression"].build,
        {"model__C": [0.1, 1.0]},
    )
    res = tune_model(spec, X, y, make_cv(Settings()), Settings(), pos_weight=3.5)
    assert set(res.cv_metrics) == {
        f"{s}_{m}"
        for s in ("mean", "std")
        for m in ("f1", "roc_auc", "pr_auc", "precision", "recall")
    }
    assert len(res.candidates) == 2 and res.candidates.loc[0, "rank_test_f1"] == 1
    assert 0.3 < res.cv_metrics["mean_f1"] < 0.8


def test_refit_with_oof_matches_cross_val_predict(small_xy) -> None:
    from sklearn.model_selection import cross_val_predict

    from src.training.threshold import refit_with_oof

    X, y = small_xy
    cv = make_cv(Settings())
    winner, oof, extra = refit_with_oof(lr_pipeline(), X, y, cv, {"lr": lr_pipeline()})
    expected = cross_val_predict(lr_pipeline(), X, y, cv=cv, method="predict_proba")[:, 1]
    np.testing.assert_allclose(oof, expected, rtol=1e-10)
    full = lr_pipeline().fit(X, y)
    np.testing.assert_allclose(winner.predict_proba(X), full.predict_proba(X), rtol=1e-10)
    np.testing.assert_allclose(extra["lr"].predict_proba(X), full.predict_proba(X), rtol=1e-10)


# ---------------------------------------------------------------- deciles/cutoff
def test_decile_table_structure() -> None:
    rng = np.random.default_rng(0)
    proba = rng.random(1000)
    y = (rng.random(1000) < proba).astype(int)  # calibrated-ish scores
    t = decile_table(y, proba)
    assert list(t["decile"]) == list(range(1, 11))
    assert (t["n"] == 100).all()
    assert t["cum_capture"].iloc[-1] == pytest.approx(1.0)
    assert t["bad_rate"].iloc[0] > t["bad_rate"].iloc[-1]
    assert t["min_score"].is_monotonic_decreasing
    assert t["recall"].iloc[-1] == pytest.approx(1.0)


def test_threshold_refinement_never_worse_than_best_decile() -> None:
    rng = np.random.default_rng(1)
    proba = rng.random(5000)
    y = (rng.random(5000) < proba**2).astype(int)
    t = decile_table(y, proba)
    choice = choose_threshold(y, proba, t)
    assert choice.f1 >= choice.decile_f1 - 1e-12
    k = choice.deciles_flagged
    assert 10 * (k - 1) <= choice.top_percent_flagged <= 10 * (k + 1)
    assert choice.flagged_share == pytest.approx(choice.top_percent_flagged / 100, abs=0.01)
    assert choice.fine_f1 >= choice.f1 - 1e-12  # unconstrained optimum is an upper bound


# ---------------------------------------------------------- skops allowlist
class _Rogue(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return X


def test_trusted_types_allowlist(small_xy) -> None:
    X, y = small_xy
    good = lr_pipeline().fit(X, y)
    types = trusted_types(good)
    assert "src.features.woe.WoEIVSelector" in types
    rogue = Pipeline([("rogue", _Rogue()), ("model", DummyClassifier())]).fit(X, y)
    with pytest.raises(RuntimeError, match="refusing to trust"):
        trusted_types(rogue)


# ------------------------------------------------------------ registry round trip
_SERVE_SIDE = """
import json, sys
import pandas as pd
from src.common.config import Settings
from tests.conftest import REPO_ROOT, TINY_THRESHOLD, lr_pipeline
from src.common.mlflow_utils import load_champion
model, info = load_champion(Settings())
rows = pd.read_json(sys.argv[1], orient="split")
proba = model.predict_proba(rows[info.input_columns].astype("float64"))[:, 1]
print("RESULT=" + json.dumps({
    "proba": proba.tolist(),
    "threshold": info.threshold,
    "version": info.version,
    "columns": info.input_columns,
    "training_imported": any(m.startswith("src.training") for m in sys.modules),
}))
"""


def test_registry_roundtrip_in_fresh_serving_process(small_xy, tiny_champion) -> None:
    """Register a real-transformer pipeline, then load @champion in a process that never
    imports src.training — exactly what the serving container does."""
    X, _ = small_xy
    pipe, version, tmp_path = tiny_champion

    # Serving sends columns in arbitrary order; the loader's signature restores it.
    rows = X.head(4)[list(reversed(X.columns))]
    payload = tmp_path / "rows.json"
    rows.to_json(payload, orient="split", index=False)
    proc = subprocess.run(
        [sys.executable, "-c", _SERVE_SIDE, str(payload)],
        cwd=tmp_path,
        env={
            "PYTHONPATH": str(REPO_ROOT),
            "PATH": "/usr/bin:/bin",
            "MLFLOW_TRACKING_URI": os.environ["MLFLOW_TRACKING_URI"],
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.split("RESULT=")[-1])
    assert result["training_imported"] is False
    assert result["version"] == version == "1"
    assert result["threshold"] == pytest.approx(TINY_THRESHOLD)
    assert result["columns"] == list(X.columns)
    np.testing.assert_allclose(result["proba"], pipe.predict_proba(X.head(4))[:, 1])
