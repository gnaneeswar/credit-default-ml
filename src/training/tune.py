"""Grid search with 3-fold stratified CV, plus per-model MLflow child runs.

Searches run with ``refit=False``: refitting every model's best candidate on the full
training set would be four sequential single-core fits, three of them discarded. Only the
winner (and LR, for the IV table) is refitted afterwards — in parallel with the winner's
out-of-fold predictions (see ``threshold.refit_with_oof``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.common.config import Settings
from src.common.logging import get_logger
from src.common.tables import log_table
from src.training.models import ModelSpec

log = get_logger(__name__)

SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
}


@dataclass
class TuneResult:
    spec: ModelSpec
    template: Pipeline  # unfitted pipeline carrying the best (CV-F1) hyper-parameters
    best_params: dict
    cv_metrics: dict[str, float]  # mean_* and std_* for the best-F1 candidate
    candidates: pd.DataFrame
    seconds: float
    best_estimator: Pipeline | None = None  # set only for refitted models
    run_id: str | None = None

    def summary_row(self) -> dict:
        row = {"model": self.spec.name, "family": self.spec.family}
        for m in SCORING:
            row[f"cv_{m}"] = self.cv_metrics[f"mean_{m}"]
        row["cv_f1_std"] = self.cv_metrics["std_f1"]
        row["n_candidates"] = self.spec.n_candidates
        row["seconds"] = self.seconds
        return row


def _candidate_table(search: GridSearchCV) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_)
    params = pd.json_normalize(res["params"]).rename(columns=lambda c: c.replace("model__", ""))
    metrics = res[[f"mean_test_{m}" for m in SCORING]].rename(
        columns=lambda c: c.replace("mean_test_", "")
    )
    table = pd.concat([params, metrics, res[["rank_test_f1", "mean_fit_time"]]], axis=1)
    return table.sort_values("rank_test_f1", ignore_index=True)


def tune_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    settings: Settings,
    pos_weight: float,
) -> TuneResult:
    log.info(
        "%s: GridSearchCV %d candidates x %d folds = %d fits (n_jobs=-1, scoring=f1)",
        spec.name,
        spec.n_candidates,
        cv.get_n_splits(),
        spec.n_candidates * cv.get_n_splits(),
    )
    base = spec.build(settings, pos_weight)
    search = GridSearchCV(
        base,
        param_grid=spec.grid,
        scoring=SCORING,
        refit=False,
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    start = time.perf_counter()
    search.fit(X, y)
    seconds = time.perf_counter() - start

    res = search.cv_results_
    i = int(np.argmin(res["rank_test_f1"]))  # best mean CV F1 (ties -> first)
    best_params = res["params"][i]
    cv_metrics = {}
    for m in SCORING:
        cv_metrics[f"mean_{m}"] = float(res[f"mean_test_{m}"][i])
        cv_metrics[f"std_{m}"] = float(res[f"std_test_{m}"][i])
    candidates = _candidate_table(search)
    log_table(log, f"{spec.name}: candidates ranked by CV F1 ({seconds:.1f}s)", candidates)
    return TuneResult(
        spec=spec,
        template=clone(base).set_params(**best_params),
        best_params=best_params,
        cv_metrics=cv_metrics,
        candidates=candidates,
        seconds=seconds,
    )


def log_child_run(result: TuneResult, experiment_id: str, cv_folds: int) -> None:
    """Log one model's search as a nested run (call inside the active parent run)."""
    import mlflow

    spec = result.spec
    with mlflow.start_run(run_name=spec.name, nested=True, experiment_id=experiment_id) as run:
        mlflow.set_tags({"model_name": spec.name, "model_family": spec.family})
        mlflow.log_params({k.replace("model__", ""): v for k, v in result.best_params.items()})
        mlflow.log_params({"n_candidates": spec.n_candidates, "cv_folds": cv_folds})
        mlflow.log_metrics({f"cv_{k}": v for k, v in result.cv_metrics.items()})
        mlflow.log_metric("search_seconds", result.seconds)
        mlflow.log_table(result.candidates, artifact_file=f"cv_candidates_{spec.name}.json")
        result.run_id = run.info.run_id
    log.info("%s: MLflow child run %s", spec.name, result.run_id)
