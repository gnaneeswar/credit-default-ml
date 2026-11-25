"""End-to-end training orchestrator (the ``Dockerfile.train`` entrypoint).

    python -m src.training.run                      # all stages
    python -m src.training.run --stages train,evaluate
    python -m src.training.run --models logistic_regression,xgboost   # quick subset

Stages: ingest -> eda -> train -> evaluate -> monitor. Each stage can run on its own: the
split is deterministic, and evaluate/monitor read the champion back from the registry.
"""

from __future__ import annotations

import argparse
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
    import mlflow

    from src.training.models import MODEL_SPECS
    from src.training.registry import log_and_register
    from src.training.threshold import choose_threshold, decile_table, oof_probabilities
    from src.training.tune import tune_model

    s = ctx.settings
    ctx.ensure_split()
    X, y = _features(ctx.train)
    log_shape(log, "X_train (raw canonical, float64)", X)
    cv = make_cv(s)
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    log.info(
        "class balance: positives=%d (%.2f%%), neg/pos=%.3f", y.sum(), 100 * y.mean(), pos_weight
    )

    exp_id = configure_tracking(s)
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

        specs = [MODEL_SPECS[m] for m in (ctx.models or MODEL_SPECS)]
        if "logistic_regression" not in {sp.name for sp in specs}:
            specs.insert(0, MODEL_SPECS["logistic_regression"])  # IV table comes from LR
        results = [tune_model(spec, X, y, cv, s, pos_weight, exp_id) for spec in specs]
        comparison = pd.DataFrame([r.summary_row() for r in results]).sort_values(
            "cv_f1", ascending=False, ignore_index=True
        )
        log_table(log, "model comparison (3-fold CV, best candidate per model)", comparison)
        best = max(results, key=lambda r: r.cv_metrics["mean_f1"])
        log.info(
            "selected %s by CV F1 %.4f ± %.4f (ROC-AUC %.4f)",
            best.spec.name,
            best.cv_metrics["mean_f1"],
            best.cv_metrics["std_f1"],
            best.cv_metrics["mean_roc_auc"],
        )

        lr = next(r for r in results if r.spec.name == "logistic_regression")
        iv = lr.best_estimator.named_steps["woe"].iv_table_
        log_table(log, f"WoE/IV ranking (LR refit on full train; drop IV < {s.iv_min})", iv)
        log.info("IV kept %d / %d features", int(iv["selected"].sum()), len(iv))

        oof = oof_probabilities(best.best_estimator, X, y, cv)
        deciles = decile_table(y, oof)
        choice = choose_threshold(y, oof, deciles)

        out = s.artifacts_dir
        frames = {
            out / "cv_comparison.csv": comparison,
            out / "iv_table.csv": iv,
            out / "train_decile_analysis.csv": deciles,
        }
        for path, frame in frames.items():
            frame.to_csv(path, index=False, lineterminator="\n")
            log_path_io(log, "write", path)
            mlflow.log_artifact(str(path))
        for path in (out / "eda_summary.json", out / "eda_report.md"):
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="eda")

        mlflow.set_tags({"best_model": best.spec.name, "best_family": best.spec.family})
        mlflow.log_params(
            {"best_model": best.spec.name, "optimal_threshold": round(choice.threshold, 6)}
        )
        mlflow.log_metrics({f"best_cv_{k}": v for k, v in best.cv_metrics.items()})
        mlflow.log_metrics({f"oof_{k}": v for k, v in choice.as_dict().items()})

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
        ):
            mlflow.log_artifact(str(s.artifacts_dir / name), artifact_path="holdout")
    log.info("holdout metrics logged to run %s", run_id)


STAGES = {
    "ingest": stage_ingest,
    "eda": stage_eda,
    "train": stage_train,
    "evaluate": stage_evaluate,
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
    ctx = Context(settings=settings, models=models)
    start = time.perf_counter()
    with stage("setup"):
        ctx.versions = log_versions(log)
        log.info("working dir %s · stages %s", Path.cwd(), stages)
    for name in stages:
        t0 = time.perf_counter()
        with stage(name):
            STAGES[name](ctx)
        ctx.timings.append({"stage": name, "seconds": time.perf_counter() - t0})
    total = time.perf_counter() - start
    ctx.timings.append({"stage": "TOTAL", "seconds": total})
    with stage("summary"):
        log_table(log, "stage timings", pd.DataFrame(ctx.timings), floatfmt=".2f")


if __name__ == "__main__":
    sys.exit(main())
