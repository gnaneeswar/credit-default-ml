"""Structured stdout logging with a per-stage context, for an interviewer watching the terminal.

Format: ``timestamp | LEVEL | module | stage | message``. The stage is set with
``with stage("train.cv"): ...`` and propagates through a ``contextvars`` variable.
"""

from __future__ import annotations

import contextvars
import importlib.metadata
import logging
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar("stage", default="-")
_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)-28s | %(stage)-14s | %(message)s"
_configured = False


_base_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs) -> logging.LogRecord:
    # Inject the stage at record creation so *every* handler (incl. pytest caplog) sees it.
    record = _base_factory(*args, **kwargs)
    record.stage = _STAGE.get()
    return record


def setup_logging(level: int = logging.INFO) -> None:
    """Idempotently route all logging to stdout with the project format."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.setLogRecordFactory(_record_factory)
    root = logging.getLogger()
    # Replace plain console handlers only; keep others (e.g. pytest caplog, uvicorn) intact.
    root.handlers[:] = [h for h in root.handlers if type(h) is not logging.StreamHandler]
    root.addHandler(handler)
    root.setLevel(level)
    # Third-party chatter stays at WARNING so our pipeline narrative is readable.
    for noisy in ("mlflow", "alembic", "urllib3", "matplotlib", "evidently", "httpx", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Tag every log line inside the block with ``name`` and log its elapsed time."""
    token = _STAGE.set(name)
    log = get_logger("pipeline.stage")
    start = time.perf_counter()
    log.info("▶ start")
    try:
        yield
    finally:
        log.info("■ done in %.2fs", time.perf_counter() - start)
        _STAGE.reset(token)


DEFAULT_LIBS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "mlflow",
    "evidently",
    "fastapi",
    "uvicorn",
    "pydantic",
)


def log_versions(logger: logging.Logger, libs: Iterable[str] = DEFAULT_LIBS) -> dict[str, str]:
    """Log installed versions of key libraries (metadata lookup; no heavy imports)."""
    versions = {"python": sys.version.split()[0]}
    for lib in libs:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "not-installed"
    logger.info("library versions: %s", ", ".join(f"{k}={v}" for k, v in versions.items()))
    return versions


def log_path_io(logger: logging.Logger, action: str, path: Path) -> None:
    """Log a file read/write with its size, e.g. ``write artifacts/x.json (1.2 KB)``."""
    size = path.stat().st_size if path.exists() else 0
    logger.info("%s %s (%s)", action, path, _human_bytes(size))


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
