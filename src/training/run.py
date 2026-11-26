"""End-to-end training orchestrator (the ``Dockerfile.train`` entrypoint).

    python -m src.training.run                      # all stages
    python -m src.training.run --stages train,evaluate
    python -m src.training.run --models logistic_regression,xgboost   # quick subset

Stages: ingest -> eda -> train -> evaluate -> monitor. Each stage can run on its own: the
split is deterministic, and evaluate/monitor read the champion back from the registry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.common.config import Settings, get_settings
from src.common.logging import get_logger, log_path_io, log_versions, stage
from src.common.mlflow_utils import configure_tracking
from src.common.tables import log_shape, log_table
from src.data.eda import run_eda
from src.data.ingest import load_dataset
from src.data.split import make_cv, split_xy, train_test

log = get_logger("pipeline")


@dataclass
class Context:
    settings: Settings
    df: pd.DataFrame | None = None
    train: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    models: list[str] | None = None  # None = all candidate models
    stages: list[str] = field(default_factory=list)
    monitor_pool: object | None = None  # pre-warmed Evidently workers (see stage_train)
    versions: dict = field(default_factory=dict)
    timings: list[dict] = field(default_factory=list)

    def ensure_split(self) -> None:
        if self.df is None:
            self.df = load_dataset(self.settings)
        if self.train is None or self.test is None:
            self.train, self.test = train_test(self.df, self.settings)


def _features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X, y = split_xy(frame)
    return X.astype("float64"), y  # served schema is float64 (robust to 20000.0 / NaN)


def stage_ingest(ctx: Context) -> None:
    ctx.df = load_dataset(ctx.settings)


def stage_eda(ctx: Context) -> None:
    ctx.ensure_split()
    run_eda(ctx.df, ctx.settings)


def stage_train(ctx: Context) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from src.training.models import MODEL_SPECS
    from src.training.registry import log_and_register
    from src.training.threshold import choose_threshold, decile_table, fit_families
    from src.training.tune import log_child_run, tune_model

    s = ctx.settings
    ctx.ensure_split()
    X, y = _features(ctx.train)
    log_shape(log, "X_train (raw canonical, float64)", X)
    cv = make_cv(s)
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    log.info(
        "class balance: positives=%d (%.2f%%), neg/pos=%.3f", y.sum(), 100 * y.mean(), pos_weight
    )

    # `import mlflow` + SQLite schema creation take ~5s of single-core work; overlap them with
    # the CPU-bound searches (which run in worker processes) instead of waiting up front.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlflow-init") as pool:
        tracking = pool.submit(configure_tracking, s)

        specs = [MODEL_SPECS[m] for m in (ctx.models or MODEL_SPECS)]
        if "logistic_regression" not in {sp.name for sp in specs}:
            specs.insert(0, MODEL_SPECS["logistic_regression"])  # IV table comes from LR
        results = [tune_model(spec, X, y, cv, s, pos_weight) for spec in specs]
        if "monitor" in ctx.stages and ctx.monitor_pool is None:
            from src.monitoring.run import start_worker_pool

            # Warm Evidently workers now; they import while the steps below run.
            ctx.monitor_pool = start_worker_pool()

        # One parallel batch: OOF folds + full refit for every family. Families are then
        # compared on the metric that is actually deployed: OOF F1 at each family's own tuned
        # cutoff (F1@0.5 is not comparable across differently class-weighted models).
        fits = fit_families({r.spec.name: r.template for r in results}, X, y, cv)
        choices = {}
        for r in results:
            fam = fits[r.spec.name]
            r.best_estimator = fam.model
            choices[r.spec.name] = choose_threshold(
                y, fam.oof, decile_table(y, fam.oof), verbose=False
            )
            r.oof_f1_tuned = choices[r.spec.name].f1
            r.oof_threshold = choices[r.spec.name].threshold
        log.info(
            "fitted %d families x (%d OOF folds + full refit) in parallel", len(results), s.n_splits
        )
        comparison = pd.DataFrame([r.summary_row() for r in results]).sort_values(
            "oof_f1_tuned", ascending=False, ignore_index=True
        )
        log_table(
            log,
            "model comparison (3-fold CV; selection = OOF F1 at each family's own cutoff)",
            comparison.drop(columns=["best_params"]),
        )
        best = max(results, key=lambda r: r.oof_f1_tuned)
        log.info(
            "selected %s by OOF F1 %.4f at its own cutoff %.4f (CV F1@0.5 %.4f ± %.4f)",
            best.spec.name,
            best.oof_f1_tuned,
            best.oof_threshold,
            best.cv_metrics["mean_f1"],
            best.cv_metrics["std_f1"],
        )

        lr = next(r for r in results if r.spec.name == "logistic_regression")
        iv = lr.best_estimator.named_steps["woe"].iv_table_
        log_table(log, f"WoE/IV ranking (LR refit on full train; drop IV < {s.iv_min})", iv)
        log.info("IV kept %d / %d features", int(iv["selected"].sum()), len(iv))

        oof = fits[best.spec.name].oof
        deciles = decile_table(y, oof)
        choice = choose_threshold(y, oof, deciles)  # verbose log for the champion
        exp_id = tracking.result()

    out = s.artifacts_dir
    frames = {
        out / "cv_comparison.csv": comparison,
        out / "iv_table.csv": iv,
        out / "train_decile_analysis.csv": deciles,
    }
    for path, frame in frames.items():
        frame.to_csv(path, index=False, lineterminator="\n")
        log_path_io(log, "write", path)

    # Report inputs: model grids (from the built specs) and OOF metrics at the threshold.
    from sklearn.metrics import average_precision_score, roc_auc_score

    reports = {
        out / "model_grids.json": {
            "cv": {
                "splitter": "StratifiedKFold",
                "n_splits": s.n_splits,
                "shuffle": True,
                "random_state": s.seed,
                "search": "GridSearchCV",
                "scoring": "f1",
                "metrics": ["f1", "roc_auc", "pr_auc", "precision", "recall"],
                "n_jobs": -1,
                "refit": False,
            },
            "selection": {
                "iv_min": s.iv_min,
                "lr": "WoE/IV filter fitted inside each CV training fold",
                "trees": "no pre-filter; implicit selection through splits",
            },
            "models": [spec.describe(s, pos_weight) for spec in specs],
        },
        out / "train_oof_metrics.json": {
            "source": f"{s.n_splits}-fold out-of-fold predictions on the training split",
            "model": best.spec.name,
            "n_rows": int(len(y)),
            **choice.as_dict(),
            "roc_auc": float(roc_auc_score(y, oof)),
            "pr_auc": float(average_precision_score(y, oof)),
        },
    }
    for path, payload in reports.items():
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        log_path_io(log, "write", path)

    import mlflow

    with mlflow.start_run(run_name="training-pipeline", experiment_id=exp_id) as parent:
        log.info("MLflow parent run %s", parent.info.run_id)
        mlflow.set_tags({"stage": "train", **{f"version.{k}": v for k, v in ctx.versions.items()}})
        mlflow.log_params(
            {
                "seed": s.seed,
                "cv_folds": s.n_splits,
                "test_size": s.test_size,
                "iv_min": s.iv_min,
                "n_train": len(X),
                "n_test": len(ctx.test),
                "scale_pos_weight": round(pos_weight, 4),
            }
        )
        for r in results:
            log_child_run(r, exp_id, s.n_splits)
        for path in [*frames, *reports]:
            mlflow.log_artifact(str(path))
        for path in (out / "eda_summary.json", out / "eda_report.md"):
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="eda")
        mlflow.set_tags({"best_model": best.spec.name, "best_family": best.spec.family})
        mlflow.log_params(
            {"best_model": best.spec.name, "optimal_threshold": round(choice.threshold, 6)}
        )
        mlflow.log_metrics({f"best_cv_{k}": v for k, v in best.cv_metrics.items()})
        mlflow.log_metrics(
            {f"oof_{k}": v for k, v in choice.as_dict().items() if isinstance(v, int | float)}
        )
        mlflow.set_tag("threshold_source", choice.source)
        log_and_register(best, X, choice, s)

    # Hand-off data for the batch client and monitoring (gitignored artifacts).
    s.holdout_dir.mkdir(parents=True, exist_ok=True)
    reference = ctx.train.copy()
    reference["prediction_proba"] = oof
    reference["prediction"] = (oof >= choice.threshold).astype(int)
    for name, frame in (("train_reference", reference), ("test", ctx.test)):
        path = s.holdout_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        log_path_io(log, "write", path)


def stage_evaluate(ctx: Context) -> None:
    import mlflow

    from src.training.evaluate import evaluate_champion

    s = ctx.settings
    ctx.ensure_split()
    X_test, y_test = _features(ctx.test)
    metrics, run_id = evaluate_champion(X_test, y_test, s)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics["at_registered_threshold"].items()})
        for name in (
            "test_metrics.json",
            "test_confusion_matrix.json",
            "test_confusion_matrix.png",
            "test_decile_analysis.csv",
            "feature_importance.csv",
            "feature_group_importance.csv",
        ):
            if (s.artifacts_dir / name).exists():
                mlflow.log_artifact(str(s.artifacts_dir / name), artifact_path="holdout")
    log.info("holdout metrics logged to run %s", run_id)


def stage_monitor(ctx: Context) -> None:
    from src.monitoring.run import run_monitoring

    run_monitoring(ctx.settings, pool=ctx.monitor_pool)


def stage_report(ctx: Context) -> None:
    from src.reporting.ppt import build_from_artifacts

    build_from_artifacts(ctx.settings)


STAGES = {
    "ingest": stage_ingest,
    "eda": stage_eda,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "monitor": stage_monitor,
    "report": stage_report,
}
DEFAULT_STAGES = list(STAGES)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Credit default training pipeline")
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help=f"comma-separated subset of {DEFAULT_STAGES} (default: all)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated subset of candidate models (default: all four)",
    )
    args = parser.parse_args(argv)
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    unknown = [x for x in stages if x not in STAGES]
    if unknown:
        parser.error(f"unknown stages {unknown}; choose from {DEFAULT_STAGES}")

    settings = get_settings()
    settings.ensure_dirs()
    from src.training.models import MODEL_SPECS

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    if models and (bad := [m for m in models if m not in MODEL_SPECS]):
        parser.error(f"unknown models {bad}; choose from {list(MODEL_SPECS)}")
    ctx = Context(settings=settings, models=models, stages=stages)
    start = time.perf_counter()
    with stage("setup"):
        ctx.versions = log_versions(log)
        log.info("working dir %s · stages %s", Path.cwd(), stages)
    try:
        for name in stages:
            t0 = time.perf_counter()
            with stage(name):
                STAGES[name](ctx)
            ctx.timings.append({"stage": name, "seconds": time.perf_counter() - t0})
    finally:
        if ctx.monitor_pool is not None:
            ctx.monitor_pool.shutdown(cancel_futures=True)
    total = time.perf_counter() - start
    ctx.timings.append({"stage": "TOTAL", "seconds": total})
    with stage("summary"):
        log_table(log, "stage timings", pd.DataFrame(ctx.timings), floatfmt=".2f")


if __name__ == "__main__":
    sys.exit(main())
