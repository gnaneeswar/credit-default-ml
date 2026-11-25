"""FastAPI prediction service.

    uvicorn src.inference.app:app --host 0.0.0.0 --port 8000

At startup the service loads ``models:/credit-default-pipeline@champion`` and its registered
decision threshold from the SQLite Model Registry via the MLflow SDK. It depends only on
``src.common`` (config, logging, registry loader) and, through the unpickled pipeline, on
``src.features`` — never on training or data code, so the serving image ships neither.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.common.config import get_settings
from src.common.logging import get_logger, log_versions
from src.common.mlflow_utils import ChampionInfo, load_champion
from src.inference.schemas import (
    HealthResponse,
    InvocationRequest,
    InvocationResponse,
    Prediction,
)

log = get_logger("inference.app")

Loader = Callable[[], tuple[Any, ChampionInfo]]


def _default_loader() -> tuple[Any, ChampionInfo]:
    return load_champion(get_settings())


def create_app(loader: Loader | None = None) -> FastAPI:
    loader = loader or _default_loader

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model, app.state.info, app.state.load_error = None, None, None
        log_versions(log, ("mlflow", "scikit-learn", "lightgbm", "xgboost", "fastapi", "pandas"))
        try:
            app.state.model, app.state.info = loader()
            log.info("serving %s", app.state.info.as_dict())
        except Exception as exc:  # keep the process up so /health can explain the failure
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("model load failed; /invocations will return 503")
        yield

    app = FastAPI(
        title="Credit Default Prediction API",
        version="1.0.0",
        description="Scores credit-card clients for next-month default risk using the "
        "MLflow-registered @champion pipeline.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default echoes the offending input back: a bad 6,000-row batch would be
        # returned in full, and a NaN in it would crash JSON encoding into a 500. Return
        # only where and why validation failed.
        errors = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        log.warning("rejected request: %d validation error(s): %s", len(errors), errors[:3])
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> JSONResponse:
        info: ChampionInfo | None = request.app.state.info
        if request.app.state.model is None or info is None:
            body = HealthResponse(
                status="unavailable", model_loaded=False, error=request.app.state.load_error
            )
            return JSONResponse(status_code=503, content=body.model_dump())
        body = HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=info.name,
            model_alias=info.alias,
            model_version=info.version,
            model_family=info.tags.get("model_family"),
            threshold=info.threshold,
            n_features=len(info.input_columns),
        )
        return JSONResponse(content=body.model_dump())

    @app.post("/invocations", response_model=InvocationResponse)
    def invocations(payload: InvocationRequest, request: Request) -> InvocationResponse:
        model, info = request.app.state.model, request.app.state.info
        if model is None or info is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        start = time.perf_counter()
        split = payload.dataframe_split

        expected = info.input_columns
        missing = [c for c in expected if c not in split.columns]
        unknown = [c for c in split.columns if c not in expected]
        if missing or unknown:
            log.warning("rejected request: missing=%s unknown=%s", missing, unknown)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "columns do not match the model signature",
                    "missing_columns": missing,
                    "unknown_columns": unknown,
                    "expected_columns": expected,
                },
            )

        # Reorder to the training-time column order (sklearn checks feature names *and*
        # order) and cast to the signed dtype (float64).
        frame = pd.DataFrame(split.data, columns=split.columns)[expected].astype("float64")
        proba = np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
        pred = (proba >= info.threshold).astype(int)

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "scored %d rows in %.1f ms · flagged %d (%.1f%%) at threshold %.4f",
            len(pred),
            elapsed_ms,
            int(pred.sum()),
            100 * pred.mean(),
            info.threshold,
        )
        return InvocationResponse(
            predictions=[
                Prediction(default_probability=float(p), prediction=int(c))
                for p, c in zip(proba, pred, strict=True)
            ],
            index=split.index,
            threshold=info.threshold,
            model_name=info.name,
            model_version=info.version,
            n_rows=len(pred),
        )

    return app


app = create_app()
