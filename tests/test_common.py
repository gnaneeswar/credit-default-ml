"""Phase 0 smoke tests: settings, path discipline, logging format."""

import logging
from pathlib import Path

import pandas as pd

from src.common.config import Settings
from src.common.logging import get_logger, log_versions, stage
from src.common.tables import render_table


def test_settings_paths_are_pathlib_and_relative_uris(tmp_path: Path) -> None:
    s = Settings(root=tmp_path)
    assert isinstance(s.raw_csv, Path)
    assert s.raw_csv.is_relative_to(tmp_path)
    assert s.tracking_uri == "sqlite:///mlruns/mlflow.db"
    assert s.artifact_root == "./mlruns"
    assert s.model_uri == "models:/credit-default-pipeline@champion"


def test_ensure_dirs_creates_output_dirs(tmp_path: Path) -> None:
    s = Settings(root=tmp_path)
    s.ensure_dirs()
    assert s.artifacts_dir.is_dir() and s.holdout_dir.is_dir() and s.mlruns_dir.is_dir()


def test_stage_context_tags_records(caplog) -> None:
    log = get_logger("test")
    with caplog.at_level(logging.INFO), stage("unit.test"):
        log.info("hello")
    assert any(getattr(r, "stage", None) == "unit.test" for r in caplog.records)


def test_log_versions_reports_python() -> None:
    versions = log_versions(get_logger("test"), libs=["pandas"])
    assert "python" in versions and versions["pandas"] != "not-installed"


def test_render_table() -> None:
    out = render_table(pd.DataFrame({"a": [1.0], "b": ["x"]}))
    assert "a" in out and "x" in out
