"""Project-wide settings. Every location is a ``pathlib.Path`` relative to the working dir.

Both Docker images use ``WORKDIR /app`` and the same relative MLflow URIs, so
artifact locations recorded by the training container resolve in the serving one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Thread-pool pinning (OMP_NUM_THREADS=1 etc.) lives in src/__init__.py so it runs first.


def _root() -> Path:
    return Path(os.environ.get("APP_ROOT", Path.cwd())).resolve()


@dataclass(frozen=True)
class Settings:
    root: Path = field(default_factory=_root)

    # --- reproducibility & modelling invariants ---
    seed: int = 42
    test_size: float = 0.20
    n_splits: int = 3
    iv_min: float = 0.02

    # --- MLflow (relative URIs on purpose: identical across local + containers) ---
    # Honours MLflow's standard MLFLOW_TRACKING_URI env var; default is the relative store.
    tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    )
    artifact_root: str = "./mlruns"
    experiment_name: str = "credit-default"
    model_name: str = "credit-default-pipeline"
    model_alias: str = "champion"

    # --- serving ---
    api_url: str = "http://localhost:8000"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_csv(self) -> Path:
        return self.data_dir / "raw" / "default_of_credit_card_clients.csv"

    @property
    def dictionary_json(self) -> Path:
        return self.data_dir / "data_dictionary.json"

    @property
    def dictionary_md(self) -> Path:
        return self.data_dir / "data_dictionary.md"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def holdout_dir(self) -> Path:
        return self.artifacts_dir / "holdout"

    @property
    def mlruns_dir(self) -> Path:
        return self.root / "mlruns"

    @property
    def model_uri(self) -> str:
        return f"models:/{self.model_name}@{self.model_alias}"

    def ensure_dirs(self) -> None:
        for d in (self.artifacts_dir, self.holdout_dir, self.mlruns_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
