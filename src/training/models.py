"""Candidate pipelines and compact hyper-parameter grids.

* Logistic Regression: clean -> engineer -> WoE/IV selection -> LR. WoE linearises each
  feature's relationship with log-odds and IV drops weak predictors, which is what a
  linear model needs.
* Trees (RF, LightGBM, XGBoost): clean -> engineer -> estimator on raw, unbinned features.
  Trees find their own split points and capture interactions; WoE binning would only
  discard resolution.

Every estimator runs single-threaded (n_jobs=1); parallelism is one process per CV fit at
the GridSearchCV level. Imbalance is handled with class weights so that F1 at the default
0.5 cutoff is a meaningful ranking criterion during the search; the operating threshold is
tuned separately on out-of-fold predictions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.common.config import Settings
from src.features.domain import DomainCleaner, DomainFeatureEngineer
from src.features.woe import WoEIVSelector


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    build: Callable[[Settings, float], Pipeline]
    grid: dict[str, list]
    display: str = ""
    preprocessing: str = ""
    imbalance: str = ""

    @property
    def n_candidates(self) -> int:
        n = 1
        for values in self.grid.values():
            n *= len(values)
        return n

    def describe(self, settings: Settings, pos_weight: float) -> dict:
        """JSON-ready description for reports, read from the *built* estimator so it cannot
        drift from the code (grid, imbalance handling and fixed hyper-parameters)."""
        model = self.build(settings, pos_weight).named_steps["model"]
        params = model.get_params()
        fixed = {
            k: params[k]
            for k in FIXED_PARAM_KEYS
            if k in params and params[k] is not None and params[k] != -1  # -1 = "no limit"
        }
        for k in self.grid:  # searched params are reported in the grid, not as fixed
            fixed.pop(k.removeprefix("model__"), None)
        return {
            "name": self.name,
            "display": self.display or self.name,
            "family": self.family,
            "estimator": type(model).__name__,
            "preprocessing": self.preprocessing,
            "imbalance": self.imbalance,
            "grid": {k.removeprefix("model__"): v for k, v in self.grid.items()},
            "n_candidates": self.n_candidates,
            "fixed_params": fixed,
        }


FIXED_PARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "max_features",
    "max_samples",
    "class_weight",
    "is_unbalance",
    "scale_pos_weight",
    "subsample",
    "colsample_bytree",
    "tree_method",
    "max_iter",
)


def _prep() -> list[tuple[str, object]]:
    return [("clean", DomainCleaner()), ("features", DomainFeatureEngineer())]


def _logreg(s: Settings, _pos_weight: float) -> Pipeline:
    return Pipeline(
        [
            *_prep(),
            ("woe", WoEIVSelector(iv_min=s.iv_min)),
            ("model", LogisticRegression(max_iter=2000, random_state=s.seed)),
        ]
    )


def _random_forest(s: Settings, _pos_weight: float) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_samples=0.5,  # each tree bootstraps half the rows: ~2x faster, more decorrelated
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=s.seed,
    )
    return Pipeline([*_prep(), ("model", rf)])


def _lightgbm(s: Settings, _pos_weight: float) -> Pipeline:
    from lightgbm import LGBMClassifier  # needs libgomp; imported only when used

    lgbm = LGBMClassifier(
        n_estimators=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        is_unbalance=True,
        n_jobs=1,
        random_state=s.seed,
        verbose=-1,
    )
    return Pipeline([*_prep(), ("model", lgbm)])


def _xgboost(s: Settings, pos_weight: float) -> Pipeline:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=200,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        n_jobs=1,
        random_state=s.seed,
    )
    return Pipeline([*_prep(), ("model", xgb)])


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "logistic_regression",
            "linear+woe",
            _logreg,
            {"model__C": [0.01, 0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
            display="Logistic Regression",
            preprocessing="clean → features → WoE/IV selection (IV ≥ 0.02, fitted per fold) → LR",
            imbalance="class_weight ∈ {None, balanced} (searched)",
        ),
        ModelSpec(
            "random_forest",
            "bagged trees",
            _random_forest,
            {"model__max_depth": [8, 12], "model__min_samples_leaf": [10, 30]},
            display="Random Forest",
            preprocessing="clean → features → trees on raw, unbinned features",
            imbalance="class_weight = balanced_subsample",
        ),
        ModelSpec(
            "lightgbm",
            "boosted trees",
            _lightgbm,
            {"model__num_leaves": [15, 31], "model__learning_rate": [0.05, 0.1]},
            display="LightGBM",
            preprocessing="clean → features → trees on raw, unbinned features",
            imbalance="is_unbalance = True",
        ),
        ModelSpec(
            "xgboost",
            "boosted trees",
            _xgboost,
            {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]},
            display="XGBoost",
            preprocessing="clean → features → trees on raw, unbinned features (hist)",
            imbalance="scale_pos_weight = neg/pos",
        ),
    )
}
