"""Stateless domain transformers shared by every model pipeline.

Neither transformer learns anything from data (``fit`` only records input columns), so
placing them inside a Pipeline cannot leak information across CV folds. They live in
``src/features`` — not ``src/training`` — because the pickled pipeline references them and
the serving image must be able to unpickle it without any training code.

Month convention: position 1 = Sep 2005 (most recent) … position 6 = Apr 2005.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]

EDUCATION_REMAP = {0: 4, 5: 4, 6: 4}  # undocumented -> 'others'
MARRIAGE_REMAP = {0: 3}  # undocumented -> 'others'

COVER_CLIP = 5.0  # payment can exceed the prior bill (prepayment); cap the long tail


def _require(X: pd.DataFrame, cols: list[str], who: str) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{who} expects a pandas DataFrame, got {type(X).__name__}")
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"{who}: missing required columns {missing}")


class _StatelessFrameTransformer(TransformerMixin, BaseEstimator):
    """Base for transformers with no learned state: fit only records the input columns."""

    required: list[str] = []  # columns transform() needs
    _added: list[str] = []  # columns transform() appends

    def fit(self, X: pd.DataFrame, y=None):
        _require(X, self.required, type(self).__name__)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        cols = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(cols + [c for c in self._added if c not in cols], dtype=object)


class DomainCleaner(_StatelessFrameTransformer):
    """Remap undocumented demographic codes; leave PAY_n behavioural codes untouched.

    * EDUCATION {0, 5, 6} -> 4 ('others'); MARRIAGE {0} -> 3 ('others').
    * PAY_n keeps -2 (no consumption), -1 (paid in full), 0 (revolving), 1..9 (months late):
      they are distinct behaviours with distinct default rates, so they are NOT collapsed.
    """

    required = ["EDUCATION", "MARRIAGE"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _require(X, self.required, type(self).__name__)
        out = X.copy()
        out["EDUCATION"] = out["EDUCATION"].replace(EDUCATION_REMAP)
        out["MARRIAGE"] = out["MARRIAGE"].replace(MARRIAGE_REMAP)
        return out


UTIL = [f"UTIL_{i}" for i in range(1, 7)]
PAY_COVER = [f"PAY_COVER_{i}" for i in range(1, 6)]
ENGINEERED = [
    *UTIL,
    "UTIL_MEAN",
    "UTIL_MAX",
    *PAY_COVER,
    "PAY_COVER_MEAN",
    "BILL_TREND",
    "PAY_AMT_TREND",
    "N_BILL_NONPOS",
    "N_DELINQ",
    "MAX_DELINQ",
    "N_REVOLVING",
    "N_INACTIVE",
    "N_PAID_FULL",
]

# x-axis for 6-month trends in chronological order (Apr=0 … Sep=5), centred.
_T = np.arange(6, dtype=float)
_T_CENTRED = _T - _T.mean()
_T_SS = float((_T_CENTRED**2).sum())


def _slope(values_recent_first: np.ndarray) -> np.ndarray:
    """Least-squares slope per row; input columns are ordered month 1 (Sep) … 6 (Apr)."""
    chronological = values_recent_first[:, ::-1]
    return (chronological * _T_CENTRED).sum(axis=1) / _T_SS


class DomainFeatureEngineer(_StatelessFrameTransformer):
    """Append credit-behaviour features; original columns are kept for downstream models.

    * ``UTIL_n``        = BILL_AMTn / LIMIT_BAL (utilisation; negative = credit balance)
    * ``PAY_COVER_n``   = PAY_AMTn / BILL_AMT(n+1): payment made in month n against the
                          prior statement. If that bill is <= 0 nothing was owed -> 1.0
                          (fully covered). Clipped to [0, 5].
    * ``BILL_TREND`` / ``PAY_AMT_TREND`` = 6-month least-squares slope / LIMIT_BAL
                          (positive = balance or payments growing towards Sep 2005)
    * ``N_BILL_NONPOS`` = months with BILL_AMT <= 0 (amount-based inactivity)
    * ``N_INACTIVE``    = months with PAY_n == -2 (status-based "no consumption").
                          Kept separate from N_BILL_NONPOS on purpose: the EDA shows -2
                          often co-occurs with a non-zero bill.
    * ``N_DELINQ`` = months with PAY_n > 0; ``MAX_DELINQ`` = worst delay (>= 0);
      ``N_REVOLVING`` = months with PAY_n == 0; ``N_PAID_FULL`` = months with PAY_n == -1
    """

    required = ["LIMIT_BAL", *PAY_STATUS, *BILL_AMT, *PAY_AMT]
    _added = ENGINEERED

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _require(X, self.required, type(self).__name__)
        limit = X["LIMIT_BAL"].to_numpy(dtype=float)
        limit = np.where(limit > 0, limit, np.nan)  # never divide by a non-positive limit
        bills = X[BILL_AMT].to_numpy(dtype=float)
        pays = X[PAY_AMT].to_numpy(dtype=float)
        status = X[PAY_STATUS].to_numpy(dtype=float)

        util = bills / limit[:, None]

        prior_bill = bills[:, 1:]  # BILL_AMT2..6 pair with PAY_AMT1..5
        owed = prior_bill > 0
        cover = np.ones_like(prior_bill)
        np.divide(pays[:, :5], prior_bill, out=cover, where=owed)
        cover = np.clip(cover, 0.0, COVER_CLIP)

        feats = {
            **{name: util[:, i] for i, name in enumerate(UTIL)},
            "UTIL_MEAN": util.mean(axis=1),
            "UTIL_MAX": util.max(axis=1),
            **{name: cover[:, i] for i, name in enumerate(PAY_COVER)},
            "PAY_COVER_MEAN": cover.mean(axis=1),
            "BILL_TREND": _slope(bills) / limit,
            "PAY_AMT_TREND": _slope(pays) / limit,
            "N_BILL_NONPOS": (bills <= 0).sum(axis=1),
            "N_DELINQ": (status > 0).sum(axis=1),
            "MAX_DELINQ": np.clip(status.max(axis=1), 0, None),
            "N_REVOLVING": (status == 0).sum(axis=1),
            "N_INACTIVE": (status == -2).sum(axis=1),
            "N_PAID_FULL": (status == -1).sum(axis=1),
        }
        engineered = pd.DataFrame(feats, index=X.index).astype(float)
        # Only a non-positive LIMIT_BAL (never observed) can yield NaN; neutralise it.
        engineered = engineered.fillna(0.0)
        return pd.concat([X, engineered], axis=1)
