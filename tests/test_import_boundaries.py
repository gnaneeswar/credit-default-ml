"""Architecture guard: serving code never depends on training or data code.

The serving image ships only src/common, src/features and src/inference, so any import of
src.training or src.data from src/inference would crash the container at runtime.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

INFERENCE_DIR = REPO_ROOT / "src" / "inference"
SERVE_SHIPS = ("src.common", "src.features", "src.inference")
FORBIDDEN = ("src.training", "src.data", "src.monitoring", "src.reporting")


def _module_of(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def _imports(path: Path, module: str | None = None) -> set[str]:
    """Absolute module names imported by ``path``, with relative imports resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = (module or _module_of(path)).rsplit(".", 1)[0]  # this module's package
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # from .x / ..x import y -> resolve against this package
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                mod = ".".join([*base, node.module] if node.module else base)
            else:
                mod = node.module or ""
            names.add(mod)
            names.update(f"{mod}.{a.name}" for a in node.names)  # `from src import training`
    return names


def test_relative_import_resolution(tmp_path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text("from ..training import run\nfrom . import app\n", encoding="utf-8")
    names = _imports(probe, module="src.inference.probe")
    assert "src.training" in names and "src.training.run" in names and "src.inference.app" in names


@pytest.mark.parametrize("path", sorted(INFERENCE_DIR.glob("*.py")), ids=lambda p: p.name)
def test_inference_modules_import_only_shipped_packages(path: Path) -> None:
    for name in _imports(path):
        assert not name.startswith(FORBIDDEN), f"{path.name} imports {name}"
        if name.startswith("src."):
            assert name.startswith(SERVE_SHIPS), f"{path.name} imports {name}"


@pytest.mark.parametrize("pkg", ["common", "features"])
def test_shared_packages_do_not_import_training(pkg: str) -> None:
    for path in sorted((REPO_ROOT / "src" / pkg).glob("*.py")):
        assert not any(n.startswith(FORBIDDEN) for n in _imports(path)), path.name


def test_app_import_does_not_load_training_modules() -> None:
    code = (
        "import sys, src.inference.app, src.inference.batch_client; "
        f"bad = [m for m in sys.modules if m.startswith({FORBIDDEN!r})]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr[-1500:]
