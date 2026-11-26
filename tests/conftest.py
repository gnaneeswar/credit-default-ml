"""Shared fixtures: a small real training sample and a tiny registered @champion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.common.config import Settings
from src.data.ingest import load_dataset
from src.data.split import split_xy, train_test
from src.features.domain import DomainCleaner, DomainFeatureEngineer
from src.features.woe import WoEIVSelector

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_THRESHOLD = 0.42


@pytest.fixture(scope="session")
def small_xy() -> tuple[pd.DataFrame, pd.Series]:
    train, _ = train_test(load_dataset(Settings(root=REPO_ROOT)))
    X, y = split_xy(train.sample(3000, random_state=7).reset_index(drop=True))
    return X.astype("float64"), y


def lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("clean", DomainCleaner()),
            ("features", DomainFeatureEngineer()),
            ("woe", WoEIVSelector()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def _register(pipe, spec_name: str, X, tmp_path, monkeypatch) -> str:
    """Register a fitted pipeline as @champion in an isolated tmp registry; returns version."""
    import mlflow

    from src.common.mlflow_utils import configure_tracking
    from src.training.models import MODEL_SPECS
    from src.training.registry import log_and_register
    from src.training.threshold import ThresholdChoice
    from src.training.tune import TuneResult

    monkeypatch.chdir(tmp_path)
    # Distinct absolute URI per test: MLflow caches stores by URI *string*, so reusing the
    # relative default across tmp dirs in one process would hit the previous test's db.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}")
    settings = Settings(root=tmp_path)
    best = TuneResult(
        spec=MODEL_SPECS[spec_name],
        template=pipe,
        best_params={},
        cv_metrics={"mean_f1": 0.5, "mean_roc_auc": 0.75, "std_f1": 0.01},
        candidates=pd.DataFrame(),
        seconds=0.0,
        best_estimator=pipe,
    )
    choice = ThresholdChoice(
        threshold=TINY_THRESHOLD,
        deciles_flagged=3,
        decile_f1=0.5,
        top_percent_flagged=27,
        f1=0.51,
        precision=0.45,
        recall=0.6,
        flagged_share=0.27,
        fine_threshold=0.41,
        fine_f1=0.52,
        f1_at_050=0.5,
    )
    exp_id = configure_tracking(settings)
    with mlflow.start_run(experiment_id=exp_id):
        return log_and_register(best, X, choice, settings)


@pytest.fixture
def tiny_champion(small_xy, tmp_path, monkeypatch):
    """Real-transformer LR pipeline registered as @champion. Returns (pipe, version, tmp)."""
    X, y = small_xy
    pipe = lr_pipeline().fit(X, y)
    return pipe, _register(pipe, "logistic_regression", X, tmp_path, monkeypatch), tmp_path


@pytest.fixture
def tiny_rf_champion(small_xy, tmp_path, monkeypatch):
    """Real tree champion (small RF) registered through the real registry path.

    It is fitted with n_jobs=4 on purpose: the registry must force n_jobs=1 before logging,
    so the served model is single-threaded (C1). Returns (pipe, version, tmp)."""
    from src.training.models import MODEL_SPECS

    X, y = small_xy
    pipe = MODEL_SPECS["random_forest"].build(Settings(), 3.5)
    pipe.set_params(model__n_estimators=20, model__max_depth=6, model__n_jobs=4).fit(X, y)
    return pipe, _register(pipe, "random_forest", X, tmp_path, monkeypatch), tmp_path
