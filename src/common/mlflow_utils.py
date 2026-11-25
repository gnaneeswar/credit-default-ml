"""MLflow helpers shared by training, evaluation, serving and monitoring.

MLflow is used strictly as a library: a SQLite tracking store + Model Registry at
``sqlite:///mlruns/mlflow.db`` with artifacts under ``./mlruns``. Both containers run with
``WORKDIR /app``, so the absolute artifact URIs recorded at training time
(``file:///app/mlruns/...``) resolve identically in the serving container.

Holdout evaluation, the FastAPI service and monitoring all load the champion through
``load_champion`` — the evaluated code path *is* the serving code path.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

from src.common.config import Settings, get_settings
from src.common.logging import get_logger

log = get_logger(__name__)

THRESHOLD_TAG = "optimal_threshold"


def quiet_mlflow() -> None:
    """Silence third-party deprecation chatter that would bury the pipeline narrative."""
    for category in (DeprecationWarning, FutureWarning, PendingDeprecationWarning):
        warnings.filterwarnings("ignore", category=category)
    warnings.filterwarnings("ignore", message=".*Integer columns.*")
    warnings.filterwarnings("ignore", message=".*Operation with range is deprecated.*")
    warnings.filterwarnings("ignore", message=".*noload.*")
    for name in ("mlflow", "alembic", "mlflow.utils.environment", "mlflow.agent"):
        logging.getLogger(name).setLevel(logging.ERROR)


def configure_tracking(settings: Settings | None = None, create: bool = True) -> str | None:
    """Point MLflow at the SQLite store; ensure the experiment exists with a portable root.

    Returns the experiment id (``None`` if it does not exist and ``create`` is False).
    Fails fast if an existing experiment's artifact root differs from ``./mlruns`` under
    the current working dir — e.g. a db trained on the host mounted into ``/app``.
    """
    import os

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    quiet_mlflow()
    settings = settings or get_settings()
    settings.mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.tracking_uri)
    expected = settings.mlruns_dir.resolve().as_uri()

    exp = mlflow.get_experiment_by_name(settings.experiment_name)
    if exp is None:
        if not create:
            return None
        exp_id = mlflow.create_experiment(settings.experiment_name, artifact_location=expected)
        log.info(
            "created experiment '%s' (id=%s) at %s", settings.experiment_name, exp_id, expected
        )
        return exp_id
    if exp.artifact_location.rstrip("/") != expected.rstrip("/"):
        raise RuntimeError(
            f"MLflow experiment '{settings.experiment_name}' stores artifacts at "
            f"{exp.artifact_location}, but this process expects {expected}. The tracking db was "
            "created under a different working directory (e.g. trained on the host, then "
            "mounted into /app). Remove ./mlruns and retrain in this environment."
        )
    log.info(
        "tracking %s · experiment '%s' (id=%s)", settings.tracking_uri, exp.name, exp.experiment_id
    )
    return exp.experiment_id


@dataclass(frozen=True)
class ChampionInfo:
    name: str
    alias: str
    version: str
    run_id: str
    threshold: float
    input_columns: list[str]
    input_types: dict[str, str]
    tags: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "model_alias": self.alias,
            "model_version": self.version,
            "run_id": self.run_id,
            "threshold": self.threshold,
            "model_family": self.tags.get("model_family"),
        }


def champion_info(settings: Settings | None = None) -> ChampionInfo:
    """Resolve ``@champion`` metadata: version, threshold tag, and the input signature."""
    import mlflow
    from mlflow import MlflowClient

    settings = settings or get_settings()
    configure_tracking(settings, create=False)
    mv = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias)
    if THRESHOLD_TAG not in mv.tags:
        raise RuntimeError(f"model version {mv.version} has no '{THRESHOLD_TAG}' tag")
    signature = mlflow.models.get_model_info(settings.model_uri).signature
    if signature is None:
        raise RuntimeError("champion model was logged without a signature")
    inputs = signature.inputs
    return ChampionInfo(
        name=settings.model_name,
        alias=settings.model_alias,
        version=str(mv.version),
        run_id=mv.run_id,
        threshold=float(mv.tags[THRESHOLD_TAG]),
        input_columns=list(inputs.input_names()),
        input_types={c.name: str(c.type) for c in inputs.inputs},
        tags=dict(mv.tags),
    )


def load_champion(settings: Settings | None = None):
    """Load ``models:/<name>@champion`` (sklearn flavor, skops format) and its metadata."""
    import mlflow.sklearn

    settings = settings or get_settings()
    info = champion_info(settings)
    log.info("loading %s (version %s, run %s)", settings.model_uri, info.version, info.run_id)
    model = mlflow.sklearn.load_model(settings.model_uri)
    log.info(
        "loaded %s · threshold=%.4f · %d input columns",
        type(model).__name__,
        info.threshold,
        len(info.input_columns),
    )
    return model, info
