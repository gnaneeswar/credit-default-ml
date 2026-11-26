"""Phase 3: tuning, decile threshold selection, skops allowlist, registry round-trip."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
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


def test_fit_families_matches_cross_val_predict_and_is_single_threaded(small_xy) -> None:
    from sklearn.model_selection import cross_val_predict

    from src.training.threshold import fit_families

    X, y = small_xy
    cv = make_cv(Settings())
    rf = MODEL_SPECS["random_forest"].build(Settings(), 3.5).set_params(model__n_estimators=10)
    fits = fit_families({"lr": lr_pipeline(), "rf": rf}, X, y, cv)
    for name, template in (("lr", lr_pipeline()), ("rf", rf)):
        expected = cross_val_predict(template, X, y, cv=cv, method="predict_proba")[:, 1]
        np.testing.assert_allclose(fits[name].oof, expected, rtol=1e-10)
        full = clone(template).fit(X, y)
        np.testing.assert_allclose(
            fits[name].model.predict_proba(X), full.predict_proba(X), rtol=1e-10
        )
    assert fits["rf"].model.named_steps["model"].n_jobs == 1  # C1: servable single-threaded


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


def test_banded_deciles_reproduce_reference_and_measure_stability() -> None:
    from src.training.threshold import banded_decile_table, population_stability_index

    rng = np.random.default_rng(3)
    proba = rng.random(4000)
    y = (rng.random(4000) < proba**2).astype(int)
    ref = decile_table(y, proba)
    same = banded_decile_table(y, proba, ref)  # a batch on its own bands == the reference
    cols = ["n", "n_default", "bad_rate", "cum_capture", "lift", "f1"]
    np.testing.assert_allclose(same[cols].to_numpy(float), ref[cols].to_numpy(float))
    assert population_stability_index(ref["share"], same["share"]) == pytest.approx(0.0)

    shifted = banded_decile_table(y, np.clip(proba + 0.2, 0, 1), ref)
    assert list(shifted["decile"]) == list(range(1, 11))  # empty bands kept, not dropped
    assert shifted["n"].sum() == len(y) and shifted.loc[9, "n"] == 0
    assert np.isfinite(shifted[["bad_rate", "lift", "cum_lift"]].to_numpy()).all()
    assert population_stability_index(ref["share"], shifted["share"]) > 0.25


def test_feature_and_group_importance(small_xy) -> None:
    from src.features.domain import FEATURE_GROUPS
    from src.training.evaluate import feature_importance, group_importance

    X, y = small_xy
    model = lr_pipeline().fit(X, y)
    fi = feature_importance(model)
    assert fi["importance_type"].iloc[0] == "|coefficient|"
    assert fi["feature"].str.startswith("WOE_").all() and "other" not in set(fi["group"])
    assert fi["importance"].is_monotonic_decreasing and list(fi["rank"]) == list(
        range(1, len(fi) + 1)
    )
    gi = group_importance(model, X, y.to_numpy(), n_repeats=2)
    total, cond = gi[gi["kind"] == "total"], gi[gi["kind"] == "conditional"]
    assert len(total) == 5 and set(total["origin"]) == {"original"}  # every raw source
    assert set(cond["origin"]) == {"engineered"} and len(cond) <= len(FEATURE_GROUPS)
    for part in (total, cond):
        assert part["auc_drop_mean"].is_monotonic_decreasing
        assert list(part["rank"]) == list(range(1, len(part) + 1))
    # Shuffling the raw repayment history through the full pipeline hurts the most.
    assert total.iloc[0]["group"] == "Repayment status" and total.iloc[0]["auc_drop_mean"] > 0


def test_trusted_type_allowlist_is_shared_and_refuses_tampering() -> None:
    from src.common import mlflow_utils
    from src.training import registry

    assert registry.is_trusted is mlflow_utils.is_trusted  # one rule for log and load time
    assert "numpy." not in mlflow_utils.SERVING_TRUSTED_PREFIXES
    mlflow_utils.check_trusted_types(["sklearn.tree._tree.Tree", "collections.OrderedDict"])
    with pytest.raises(RuntimeError, match="untrusted"):
        mlflow_utils.check_trusted_types(["sklearn.tree._tree.Tree", "builtins.eval"])
    with pytest.raises(RuntimeError, match="untrusted"):
        mlflow_utils.check_trusted_types(["collections.abc.Callable"])  # exact match only


@pytest.mark.parametrize("family", list(MODEL_SPECS))
def test_every_family_serialises_within_the_allowlist(small_xy, family: str) -> None:
    """Each candidate family must be registrable: a family-specific persisted type (e.g.
    LightGBM's OrderedDict) must not surface only on the day that family wins."""
    X, y = small_xy
    pipe = MODEL_SPECS[family].build(Settings(), 3.5)
    if "model__n_estimators" in pipe.get_params():
        pipe.set_params(model__n_estimators=5)
    trusted_types(pipe.fit(X, y))  # raises if any persisted type is not allowlisted
