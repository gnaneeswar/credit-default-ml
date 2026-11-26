"""Run monitoring scenarios against the registered @champion.

    uv run python -m src.monitoring.run

Inputs (written by training): ``artifacts/holdout/train_reference.parquet`` (train rows +
OOF predictions) and ``artifacts/holdout/test.parquet``. Outputs:
``artifacts/monitoring_report.html`` (reference vs holdout), one
``monitoring_report_<scenario>.html`` per simulated scenario, and
``artifacts/monitoring_summary.json`` — all logged to an MLflow ``monitoring`` run.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from src.common.config import Settings, get_settings
from src.common.logging import get_logger, log_path_io, stage
from src.common.mlflow_utils import configure_tracking, load_champion
from src.common.tables import log_table
from src.data.schema import FEATURES, TARGET
from src.monitoring import drift

log = get_logger(__name__)

SCENARIOS = {
    "holdout": "real 20% test set scored by @champion",
    "covariate_drift": "simulated downturn: LIMIT_BAL x0.6, PAY_n +1 month, BILL_AMT x1.4",
    "concept_drift": f"features unchanged, {drift.CONCEPT_FLIP_RATE:.0%} of labels flipped",
}


def _html_name(scenario: str) -> str:
    return (
        "monitoring_report.html" if scenario == "holdout" else f"monitoring_report_{scenario}.html"
    )


def _warm_worker() -> None:
    """Pool initializer: pay Evidently's ~5s import once per worker, ahead of time."""
    import warnings

    warnings.filterwarnings("ignore")
    import evidently.presets  # noqa: F401

    import src.monitoring.drift  # noqa: F401


def _ping() -> int:
    return os.getpid()


def start_worker_pool(n_workers: int = len(SCENARIOS)) -> ProcessPoolExecutor:
    """Start (and warm) the scenario worker processes without waiting for them.

    The training orchestrator calls this as soon as the model searches finish, so the
    Evidently import overlaps the final refits, registration and holdout evaluation.
    """
    pool = ProcessPoolExecutor(
        max_workers=n_workers, mp_context=get_context("spawn"), initializer=_warm_worker
    )
    for _ in range(n_workers):
        pool.submit(_ping)  # workers spawn lazily; force them to start warming now
    log.info("started %d warm monitoring worker processes", n_workers)
    return pool


def run_monitoring(
    settings: Settings | None = None,
    model=None,
    info=None,
    pool: ProcessPoolExecutor | None = None,
) -> dict:
    settings = settings or get_settings()
    if model is None or info is None:
        model, info = load_champion(settings)

    def predict(X: pd.DataFrame):
        return model.predict_proba(X[info.input_columns])[:, 1]

    ref_path = settings.holdout_dir / "train_reference.parquet"
    test_path = settings.holdout_dir / "test.parquet"
    for p in (ref_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run the training stage first")
        log_path_io(log, "read", p)
    reference = pd.read_parquet(ref_path)
    test = pd.read_parquet(test_path)
    log.info("reference %s (OOF predictions) · current test %s", reference.shape, test.shape)

    baseline = drift.reference_quality(reference)
    log.info("reference (OOF) quality: %s", {k: round(v, 4) for k, v in baseline.items()})

    X_test, y_test = test[FEATURES], test[TARGET]
    batches = {
        "holdout": drift.score_batch(X_test, y_test, predict, info.threshold),
        "covariate_drift": drift.score_batch(
            drift.simulate_covariate_drift(X_test), y_test, predict, info.threshold
        ),
        "concept_drift": drift.score_batch(
            X_test, drift.simulate_concept_drift(y_test, settings.seed), predict, info.threshold
        ),
    }

    # Scenarios are independent: one Evidently report per worker process (GIL-bound work,
    # so threads would not help). A pre-warmed pool from the orchestrator is reused if given.
    own_pool = pool is None
    pool = pool or start_worker_pool()
    t0 = time.perf_counter()
    try:
        futures = [
            pool.submit(
                drift.evaluate_scenario,
                name,
                SCENARIOS[name],
                reference,
                batch,
                info.threshold,
                baseline,
                settings.artifacts_dir / _html_name(name),
            )
            for name, batch in batches.items()
        ]
        results = [f.result() for f in futures]
    finally:
        if own_pool:
            pool.shutdown()
    log.info("%d Evidently reports in %.1fs (parallel)", len(results), time.perf_counter() - t0)
    for r in results:
        drift.log_scenario(r)

    summary = {
        "model": info.as_dict(),
        "reference": {"n_rows": int(len(reference)), "source": "train OOF", **baseline},
        "alert_policy": {
            "drift_share_alert": drift.DRIFT_SHARE_ALERT,
            "f1_drop_alert": drift.F1_DROP_ALERT,
            "accuracy_drop_alert": drift.ACCURACY_DROP_ALERT,
            "score_psi_alert": drift.SCORE_PSI_ALERT,
        },
        "scenarios": results,
    }
    table = pd.DataFrame(
        [
            {
                "scenario": r["scenario"],
                "rows": r["n_rows"],
                "flagged": r["flagged_share"],
                "drift_share": r["drift_share"],
                "drifted_cols": r["drifted_count"],
                "score_psi": r["score_psi"],
                "f1": r["classification"].get("f1"),
                "Δf1": -r["f1_drop"],
                "accuracy": r["classification"].get("accuracy"),
                "Δacc": -r["accuracy_drop"],
                "drift_alert": r["alerts"]["data_drift"],
                "perf_alert": r["alerts"]["f1_drop"] or r["alerts"]["accuracy_drop"],
                "score_alert": r["alerts"]["score_drift"],
            }
            for r in results
        ]
    )
    log_table(log, "monitoring summary (vs reference OOF baseline)", table)

    out = settings.artifacts_dir / "monitoring_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    log_path_io(log, "write", out)
    _log_to_mlflow(
        settings,
        info,
        results,
        [out] + [Path(r["html"]) for r in results],
    )
    return summary


def _log_to_mlflow(settings: Settings, info, results, paths) -> None:
    import mlflow

    exp_id = configure_tracking(settings)
    with mlflow.start_run(run_name="monitoring", experiment_id=exp_id) as run:
        mlflow.set_tags(
            {"stage": "monitor", "model_version": info.version, "champion_run_id": info.run_id}
        )
        for s in results:
            n = s["scenario"]
            mlflow.log_metrics(
                {
                    f"{n}_drift_share": s["drift_share"],
                    f"{n}_drifted_count": s["drifted_count"],
                    f"{n}_score_psi": s["score_psi"],
                    f"{n}_f1": s["classification"].get("f1", float("nan")),
                    f"{n}_accuracy": s["classification"].get("accuracy", float("nan")),
                    f"{n}_alert": float(any(s["alerts"].values())),
                }
            )
        for p in paths:
            mlflow.log_artifact(str(p), artifact_path="monitoring")
    log.info("monitoring logged to MLflow run %s", run.info.run_id)


def main() -> None:
    with stage("monitor"):
        run_monitoring()


if __name__ == "__main__":
    main()
