"""Request/response contracts for the prediction API.

``POST /invocations`` accepts the MLflow-style ``dataframe_split`` payload:

    {"dataframe_split": {"columns": ["LIMIT_BAL", ...], "data": [[20000, ...], ...]}}

Values must be finite JSON numbers (ints or floats). Strings, booleans, nulls, NaN and
Infinity are rejected with 422 — the training data has no missing values, so the contract
is explicit rather than silently imputing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

MAX_ROWS = 50_000
MAX_ABS = 1e15  # checked before float conversion: huge ints must be a 422, not an OverflowError


@dataclass(frozen=True)
class Rule:
    lo: float
    hi: float
    integral: bool = False
    description: str = ""


# Business contract per column (the API may not import src.data, so it lives here). Values
# outside these ranges are rejected with 422 rather than silently scored.
_PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
FEATURE_RULES: dict[str, Rule] = {
    "LIMIT_BAL": Rule(1, 1e9, description="credit limit: 1 .. 1e9"),
    "SEX": Rule(1, 2, integral=True, description="1 or 2"),
    "EDUCATION": Rule(0, 6, integral=True, description="integer code 0..6"),
    "MARRIAGE": Rule(0, 3, integral=True, description="integer code 0..3"),
    "AGE": Rule(18, 100, integral=True, description="integer age 18..100"),
    **{c: Rule(-2, 9, integral=True, description="integer status -2..9") for c in _PAY_STATUS},
    **{f"BILL_AMT{i}": Rule(-1e9, 1e9, description="|bill| <= 1e9") for i in range(1, 7)},
    **{f"PAY_AMT{i}": Rule(0, 1e9, description="payment: 0 .. 1e9") for i in range(1, 7)},
}


def rule_violations(frame: pd.DataFrame, limit: int = 20) -> tuple[int, list[dict]]:
    """Vectorised business-range check. Returns (total violations, first ``limit`` details)."""
    total, details = 0, []
    for col, rule in FEATURE_RULES.items():
        if col not in frame.columns:
            continue
        v = frame[col].to_numpy(dtype=float)
        bad = (v < rule.lo) | (v > rule.hi)
        if rule.integral:
            bad |= v != np.round(v)
        idx = np.flatnonzero(bad)
        total += len(idx)
        for i in idx[: max(0, limit - len(details))]:
            details.append(
                {"row": int(i), "column": col, "value": float(v[i]), "rule": rule.description}
            )
    return total, details


Number = Annotated[StrictInt | StrictFloat, Field(description="finite numeric feature value")]


class DataFrameSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(min_length=1, description="feature names, any order")
    data: list[list[Number]] = Field(min_length=1, max_length=MAX_ROWS)
    index: list[int | str] | None = Field(default=None, description="optional row ids")

    @model_validator(mode="after")
    def _check_shape(self) -> DataFrameSplit:
        if len(set(self.columns)) != len(self.columns):
            dupes = sorted({c for c in self.columns if self.columns.count(c) > 1})
            raise ValueError(f"duplicate columns: {dupes}")
        width = len(self.columns)
        for i, row in enumerate(self.data):
            if len(row) != width:
                raise ValueError(f"row {i} has {len(row)} values, expected {width}")
            if any(abs(v) > MAX_ABS for v in row):
                raise ValueError(f"row {i} has a value with magnitude above {MAX_ABS:g}")
            if not all(math.isfinite(v) for v in row):
                raise ValueError(f"row {i} contains NaN or Infinity")
        if self.index is not None and len(self.index) != len(self.data):
            raise ValueError(f"index has {len(self.index)} ids for {len(self.data)} rows")
        return self


class InvocationRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "dataframe_split": {
                    "columns": ["LIMIT_BAL", "SEX", "EDUCATION", "..."],
                    "data": [[20000, 2, 2, "..."]],
                }
            }
        },
    )

    dataframe_split: DataFrameSplit


class Prediction(BaseModel):
    default_probability: float = Field(ge=0.0, le=1.0)
    prediction: int = Field(description="1 = predicted default at the registered threshold")


class InvocationResponse(BaseModel):
    predictions: list[Prediction]
    index: list[int | str] | None = None
    threshold: float
    model_name: str
    model_version: str
    n_rows: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_alias: str | None = None
    model_version: str | None = None
    model_family: str | None = None
    threshold: float | None = None
    n_features: int | None = None
    error: str | None = None
