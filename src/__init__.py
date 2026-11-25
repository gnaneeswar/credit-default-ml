"""Credit-default pipeline package.

Runtime environment is pinned here because a package ``__init__`` executes before any
module inside it — so ``python -m src.training.run`` and ``uvicorn src.inference.app:app``
both get these settings before numpy / sklearn / lightgbm / xgboost load their native
thread pools (import sorting can't reorder this).
"""

import os

# Parallelism lives at the GridSearchCV level (n_jobs=-1, one process per fit). Nested
# OpenMP/BLAS threads inside each fit would oversubscribe CPUs (worst in WSL2/Docker).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# Quiet, offline-friendly MLflow: no progress bars, no usage telemetry.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
