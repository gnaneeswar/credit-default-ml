"""Request/response contracts for the prediction API.

``POST /invocations`` accepts the MLflow-style ``dataframe_split`` payload:

    {"dataframe_split": {"columns": ["LIMIT_BAL", ...], "data": [[20000, ...], ...]}}

Values must be finite JSON numbers (ints or floats). Strings, booleans, nulls, NaN and
Infinity are rejected with 422 — the training data has no missing values, so the contract
is explicit rather than silently imputing.
"""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

MAX_ROWS = 50_000

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
