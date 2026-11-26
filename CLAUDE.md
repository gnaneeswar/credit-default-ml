# CLAUDE.md

On session start: read `PROGRESS.md` (current phase, next steps). Full plan lives there.

Invariants:
- Python 3.12 via `uv` only (`uv sync`, `uv run ...`, `uv add`); never pip.
- `pathlib.Path` for all file ops — no `os.path`, no string path joins (ruff `PTH` enforces).
- LF line endings everywhere.
- Never commit `.venv/`, `mlruns/`, `*.db*`, `artifacts/*`, `.env*`.
- MLflow: library only — `sqlite:///mlruns/mlflow.db`, artifacts `./mlruns`, model `models:/credit-default-pipeline@champion`. No `mlflow models serve/build-docker`.
- Serving = FastAPI (`src/inference/`), which must never import `src.training`. Custom transformers live in `src/features/`.
- All fitted transforms stay inside sklearn Pipelines (leak-free CV).
- Canonical status columns are `PAY_1`..`PAY_6` (UCI's `PAY_0` is renamed to `PAY_1` on load); all ratios divide by `LIMIT_BAL`, never by a bill.
- Finish a phase with the `/checkpoint` skill.
