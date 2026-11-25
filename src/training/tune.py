"""Grid search with 3-fold stratified CV and per-model MLflow child runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
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
    best_estimator: Pipeline
    best_params: dict
    cv_metrics: dict[str, float]  # mean_* and std_* for the refit (best-F1) candidate
    candidates: pd.DataFrame
    seconds: float
    run_id: str | None = None
    extra: dict = field(default_factory=dict)

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
    experiment_id: str | None = None,
) -> TuneResult:
    import mlflow

    log.info(
        "%s: GridSearchCV %d candidates x %d folds = %d fits (n_jobs=-1, scoring=f1)",
        spec.name,
        spec.n_candidates,
        cv.get_n_splits(),
        spec.n_candidates * cv.get_n_splits(),
    )
    search = GridSearchCV(
        spec.build(settings, pos_weight),
        param_grid=spec.grid,
        scoring=SCORING,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    start = time.perf_counter()
    search.fit(X, y)
    seconds = time.perf_counter() - start

    i = search.best_index_
    cv_metrics = {}
    for m in SCORING:
        cv_metrics[f"mean_{m}"] = float(search.cv_results_[f"mean_test_{m}"][i])
        cv_metrics[f"std_{m}"] = float(search.cv_results_[f"std_test_{m}"][i])
    candidates = _candidate_table(search)
    log_table(log, f"{spec.name}: candidates ranked by CV F1 ({seconds:.1f}s)", candidates)

    result = TuneResult(
        spec=spec,
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        cv_metrics=cv_metrics,
        candidates=candidates,
        seconds=seconds,
    )
    if experiment_id is not None:
        with mlflow.start_run(run_name=spec.name, nested=True, experiment_id=experiment_id) as run:
            mlflow.set_tags({"model_name": spec.name, "model_family": spec.family})
            mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_params({"n_candidates": spec.n_candidates, "cv_folds": cv.get_n_splits()})
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
            mlflow.log_metric("search_seconds", seconds)
            mlflow.log_table(candidates, artifact_file=f"cv_candidates_{spec.name}.json")
            result.run_id = run.info.run_id
        log.info("%s: MLflow child run %s", spec.name, result.run_id)
    return result
