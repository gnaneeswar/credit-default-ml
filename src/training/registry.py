"""Log the winning pipeline, register it, and point the ``@champion`` alias at it.

Serialization: MLflow's sklearn flavor stores the model with **skops**, which refuses to
load any type not explicitly trusted (no arbitrary code execution on load, unlike pickle).
The trusted list is derived from the fitted pipeline, checked against an allowlist of
module prefixes, and stored in the MLmodel file so the serving side loads it unchanged.
"""

from __future__ import annotations

import importlib.metadata
import time

import pandas as pd
from sklearn.pipeline import Pipeline

from src.common.config import Settings
from src.common.logging import get_logger
from src.common.mlflow_utils import THRESHOLD_TAG, is_trusted
from src.training.threshold import ThresholdChoice
from src.training.tune import TuneResult

log = get_logger(__name__)

# The allowlist lives in src.common.mlflow_utils and is shared with the serving loader.
_RUNTIME_PACKAGES = ("scikit-learn", "pandas", "numpy", "skops", "lightgbm", "xgboost")


def trusted_types(model: Pipeline) -> list[str]:
    import skops.io as sio

    types = sorted(sio.get_untrusted_types(data=sio.dumps(model)))
    rejected = [t for t in types if not is_trusted(t)]
    if rejected:
        raise RuntimeError(f"refusing to trust unexpected types in the pipeline: {rejected}")
    log.info("skops trusted types (%d): %s", len(types), ", ".join(types))
    return types


def _pip_requirements() -> list[str]:
    # Explicit pins: faster than MLflow's import-capturing inference and reproducible.
    return [f"{p}=={importlib.metadata.version(p)}" for p in _RUNTIME_PACKAGES]


def log_and_register(
    best: TuneResult,
    X_train: pd.DataFrame,
    choice: ThresholdChoice,
    settings: Settings,
) -> str:
    """Log the model in the active (parent) run, register it, alias it. Returns the version."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    t0 = time.perf_counter()
    model = best.best_estimator
    # Served single-threaded: a multi-threaded estimator would spawn a thread pool per request
    # inside uvicorn (C1). Enforced here regardless of how the model was fitted.
    if model.get_params().get("model__n_jobs") not in (None, 1):
        model.set_params(model__n_jobs=1)
    # Raw canonical columns, as the API receives them (before feature engineering).
    example = X_train.head(5)
    signature = infer_signature(example, model.predict_proba(example)[:, 1])
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=example,
        skops_trusted_types=trusted_types(model),
        pip_requirements=_pip_requirements(),
        metadata={THRESHOLD_TAG: choice.threshold, "model_family": best.spec.family},
    )
    t1 = time.perf_counter()
    log.info("logged model %s (%s) in %.1fs", info.model_uri, best.spec.name, t1 - t0)

    mv = mlflow.register_model(info.model_uri, settings.model_name)
    client = MlflowClient()
    tags = {
        THRESHOLD_TAG: repr(float(choice.threshold)),  # full precision
        "model_name": best.spec.name,
        "model_family": best.spec.family,
        "cv_f1": f"{best.cv_metrics['mean_f1']:.4f}",
        "cv_roc_auc": f"{best.cv_metrics['mean_roc_auc']:.4f}",
        "oof_f1_at_threshold": f"{choice.f1:.4f}",
        "deciles_flagged": str(choice.deciles_flagged),
        "top_percent_flagged": str(choice.top_percent_flagged),
        "best_params": str({k.replace("model__", ""): v for k, v in best.best_params.items()}),
    }
    for k, v in tags.items():
        client.set_model_version_tag(settings.model_name, mv.version, k, v)
    client.update_model_version(
        settings.model_name,
        mv.version,
        description=(
            f"{best.spec.name} pipeline (clean -> features -> model); CV F1 "
            f"{best.cv_metrics['mean_f1']:.4f}; decision threshold {choice.threshold:.4f} "
            f"(flag riskiest {choice.top_percent_flagged}% by OOF score; best decile k="
            f"{choice.deciles_flagged})."
        ),
    )
    client.set_registered_model_alias(settings.model_name, settings.model_alias, mv.version)
    log.info(
        "registered %s version %s and set alias @%s (threshold=%.4f) in %.1fs",
        settings.model_name,
        mv.version,
        settings.model_alias,
        choice.threshold,
        time.perf_counter() - t1,
    )
    return str(mv.version)
