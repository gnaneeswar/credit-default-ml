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

PAY_STATUS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]

EDUCATION_REMAP = {0: 4, 5: 4, 6: 4}  # undocumented -> 'others'
MARRIAGE_REMAP = {0: 3}  # undocumented -> 'others'

COVER_CLIP = 5.0
_FLOAT32_MAX = float(
    np.finfo(np.float32).max
)  # payment can exceed the prior bill (prepayment); cap the long tail


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

# Billing-cycle accounting signals (aggregates + most recent month).
ACCOUNTING = [
    "HAS_CREDIT_BAL_1",
    "N_CREDIT_BAL",
    "MIN_CREDIT_BAL_RATIO",
    "OVER_LIMIT_1",
    "N_OVER_LIMIT",
    "MAX_OVER_LIMIT_EXCESS",
    "PAYDOWN_1",
    "N_PAYDOWN",
    "MAX_PAYDOWN_EXCESS",
    "SPEND_1",
    "SPEND_MEAN_3",
    "SPEND_MEAN_5",
]
# Delinquency trajectory / roll-rate signals over the 5 transitions PAY_6 -> ... -> PAY_1.
ROLL_RATE = [
    "CURED_RECENT",
    "N_CURES",
    "ROLL_FWD_RECENT",
    "N_ROLL_FWD",
    "MAX_ROLL_FWD_STEP",
    "N_DORMANT_TO_REVOLVER",
    "ROLL_BACK_RECENT",
    "N_ROLL_BACK",
    "STAGNANT_RECENT",
    "N_STAGNANT",
    "MAX_DELINQ_STREAK",
]
ENGINEERED = [*ENGINEERED, *ACCOUNTING, *ROLL_RATE]

# Feature catalogue for reports: (group, features, description). Covers ENGINEERED exactly
# once (tested). Ratios are over LIMIT_BAL unless stated otherwise.
FEATURE_GROUPS: list[tuple[str, list[str], str]] = [
    ("Utilisation", [*UTIL, "UTIL_MEAN", "UTIL_MAX"], "BILL_AMT_n / LIMIT_BAL; 6-month mean, max"),
    (
        "Payment coverage",
        [*PAY_COVER, "PAY_COVER_MEAN"],
        "PAY_AMT_n / prior bill BILL_AMT_(n+1), clipped 0–5; mean",
    ),
    ("6-month trends", ["BILL_TREND", "PAY_AMT_TREND"], "Least-squares slope of bills / payments"),
    (
        "Status counts",
        ["N_DELINQ", "MAX_DELINQ", "N_REVOLVING", "N_PAID_FULL", "N_INACTIVE", "N_BILL_NONPOS"],
        "Months late, worst delay, revolving, paid in full, inactive, bill ≤ 0",
    ),
    (
        "Credit balance",
        ["HAS_CREDIT_BAL_1", "N_CREDIT_BAL", "MIN_CREDIT_BAL_RATIO"],
        "Bill < 0 (overpayment / refund): latest flag, count, deepest credit",
    ),
    (
        "Over limit",
        ["OVER_LIMIT_1", "N_OVER_LIMIT", "MAX_OVER_LIMIT_EXCESS"],
        "Bill > limit: latest flag, count, max excess",
    ),
    (
        "Intra-cycle paydown",
        ["PAYDOWN_1", "N_PAYDOWN", "MAX_PAYDOWN_EXCESS"],
        "Paid > bill: latest flag, count, max excess",
    ),
    (
        "Implied spend",
        ["SPEND_1", "SPEND_MEAN_3", "SPEND_MEAN_5"],
        "BILL_n − (BILL_(n+1) − PAY_AMT_n): latest, 3- and 5-month means",
    ),
    ("Cure", ["CURED_RECENT", "N_CURES"], "PAY_(n+1) ≥ 2 → PAY_n ≤ 0: latest flag, history count"),
    (
        "Roll forward",
        ["ROLL_FWD_RECENT", "N_ROLL_FWD", "MAX_ROLL_FWD_STEP", "N_DORMANT_TO_REVOLVER"],
        "Worsening (cur ≥ 1): latest flag; history count, max step; −2 → 0 count",
    ),
    (
        "Roll backward",
        ["ROLL_BACK_RECENT", "N_ROLL_BACK"],
        "Improving while still past due: latest flag, history count",
    ),
    (
        "Stagnation",
        ["STAGNANT_RECENT", "N_STAGNANT", "MAX_DELINQ_STREAK"],
        "Same status ≥ 2: latest flag, history count; longest run of PAY ≥ 2",
    ),
]
FEATURE_TO_GROUP: dict[str, str] = {f: g for g, feats, _ in FEATURE_GROUPS for f in feats}

# x-axis for 6-month trends in chronological order (Apr=0 … Sep=5), centred.
_T = np.arange(6, dtype=float)
_T_CENTRED = _T - _T.mean()
_T_SS = float((_T_CENTRED**2).sum())


def _longest_run(mask_chronological: np.ndarray) -> np.ndarray:
    """Length of the longest run of consecutive True values per row."""
    run = np.zeros(mask_chronological.shape[0])
    best = np.zeros_like(run)
    for j in range(mask_chronological.shape[1]):
        run = np.where(mask_chronological[:, j], run + 1, 0)
        best = np.maximum(best, run)
    return best


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
    * ``N_INACTIVE``    = months with PAY_n == -2 (status-based "inactive cycle").
                          Kept separate from N_BILL_NONPOS on purpose: the EDA shows -2
                          often co-occurs with a non-zero bill.
    * ``N_DELINQ`` = months with PAY_n > 0; ``MAX_DELINQ`` = worst delay (>= 0);
      ``N_REVOLVING`` = months with PAY_n == 0; ``N_PAID_FULL`` = months with PAY_n == -1

    Billing-cycle accounting (every ratio is over LIMIT_BAL; nothing divides by a bill, and
    negative bills are never clipped in signed features):

    * credit balance (low risk): ``HAS_CREDIT_BAL_1`` (BILL_AMT1 < 0), ``N_CREDIT_BAL``,
      ``MIN_CREDIT_BAL_RATIO`` = most negative min(BILL_AMT_n, 0) / LIMIT_BAL (<= 0)
    * over limit (distress): ``OVER_LIMIT_1`` (BILL_AMT1 > LIMIT_BAL), ``N_OVER_LIMIT``,
      ``MAX_OVER_LIMIT_EXCESS`` = max(BILL_AMT_n - LIMIT_BAL, 0) / LIMIT_BAL
    * intra-cycle paydown (liquidity): ``PAYDOWN_1`` (PAY_AMT1 > BILL_AMT1), ``N_PAYDOWN``,
      ``MAX_PAYDOWN_EXCESS`` = max(PAY_AMT_n - BILL_AMT_n, 0) / LIMIT_BAL
    * implied spend_n = BILL_AMT_n - (BILL_AMT_(n+1) - PAY_AMT_n), n = 1..5, over LIMIT_BAL:
      ``SPEND_1`` (latest), ``SPEND_MEAN_3`` / ``SPEND_MEAN_5`` (rolling means that smooth
      calendar-month vs statement-cutoff payment timing; sign preserved)

    Roll rates over consecutive transitions prev = PAY_(n+1) -> cur = PAY_n, moving forward
    in time. ``*_RECENT`` = the latest transition (PAY_2 -> PAY_1); ``N_*`` counts and
    ``MAX_ROLL_FWD_STEP`` cover only the 4 history transitions (PAY_6 -> ... -> PAY_2), whose
    status coding is consistent (code 1 appears almost only in PAY_1):

    * cure: prev >= 2 and cur <= 0 -> ``CURED_RECENT``, ``N_CURES``. Deliberately requires
      prev >= 2: recovering from a single missed due date (1 -> <= 0) is not counted as a cure
    * roll forward (worsening): cur > prev and cur >= 1 -> ``ROLL_FWD_RECENT``, ``N_ROLL_FWD``,
      ``MAX_ROLL_FWD_STEP`` = max(cur - max(prev, 0)). Moves among non-delinquent codes are
      not worsening; ``-2 -> 2`` is a step of 2 missed due dates, not 4.
    * dormant -> revolver activation: prev == -2 and cur == 0 -> ``N_DORMANT_TO_REVOLVER``
      (no recent flag: PAY_2 = -2 -> PAY_1 = 0 never occurs in the data)
    * roll backward (improving while past due): cur < prev, both > 0 -> ``ROLL_BACK_RECENT``,
      ``N_ROLL_BACK``
    * stagnation (chronic delay): cur == prev >= 2 -> ``STAGNANT_RECENT``, ``N_STAGNANT``,
      ``MAX_DELINQ_STREAK`` = longest run of consecutive months with PAY >= 2
    """

    required = ["LIMIT_BAL", *PAY_STATUS, *BILL_AMT, *PAY_AMT]
    _added = ENGINEERED

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _require(X, self.required, type(self).__name__)
        raw = X[self.required].to_numpy(dtype=float)
        ok = np.isfinite(raw) & (np.abs(raw) <= _FLOAT32_MAX)  # trees cast inputs to float32
        if not ok.all():
            bad = [c for c, good in zip(self.required, ok.all(0), strict=True) if not good]
            raise ValueError(
                f"DomainFeatureEngineer: non-finite or beyond-float32 input values in {bad}"
            )
        limit = X["LIMIT_BAL"].to_numpy(dtype=float)
        # Defence in depth: the API rejects LIMIT_BAL < 1, so a non-positive limit can only
        # come from a direct caller; its ratio features are set to 0 rather than +/-inf.
        limit = np.where(limit > 0, limit, np.nan)
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
            **self._accounting(bills, pays, limit),
            **self._roll_rates(status),
        }
        engineered = pd.DataFrame(feats, index=X.index)[ENGINEERED].astype(float)
        engineered = engineered.fillna(0.0)  # only the non-positive-limit guard yields NaN here
        values = engineered.to_numpy()
        if not np.isfinite(values).all() or np.abs(values).max(initial=0) > _FLOAT32_MAX:
            # e.g. a tiny positive LIMIT_BAL: ratios overflow the float32 that trees use.
            raise ValueError("DomainFeatureEngineer: engineered features out of float32 range")
        return pd.concat([X, engineered], axis=1)

    @staticmethod
    def _accounting(bills: np.ndarray, pays: np.ndarray, limit: np.ndarray) -> dict:
        lim = limit[:, None]
        implied_spend = bills[:, :5] - (bills[:, 1:] - pays[:, :5])  # n = 1..5
        return {
            "HAS_CREDIT_BAL_1": bills[:, 0] < 0,
            "N_CREDIT_BAL": (bills < 0).sum(axis=1),
            "MIN_CREDIT_BAL_RATIO": np.minimum(bills, 0).min(axis=1) / limit,
            "OVER_LIMIT_1": bills[:, 0] > limit,
            "N_OVER_LIMIT": (bills > lim).sum(axis=1),
            "MAX_OVER_LIMIT_EXCESS": np.maximum(bills - lim, 0).max(axis=1) / limit,
            "PAYDOWN_1": pays[:, 0] > bills[:, 0],
            "N_PAYDOWN": (pays > bills).sum(axis=1),
            "MAX_PAYDOWN_EXCESS": np.maximum(pays - bills, 0).max(axis=1) / limit,
            "SPEND_1": implied_spend[:, 0] / limit,
            "SPEND_MEAN_3": implied_spend[:, :3].mean(axis=1) / limit,
            "SPEND_MEAN_5": implied_spend.mean(axis=1) / limit,
        }

    @staticmethod
    def _roll_rates(status_recent_first: np.ndarray) -> dict:
        chrono = status_recent_first[:, ::-1]  # PAY_6 (Apr) ... PAY_1 (Sep)
        prev, cur = chrono[:, :-1], chrono[:, 1:]  # 5 transitions; last = PAY_2 -> PAY_1
        cure = (prev >= 2) & (cur <= 0)
        roll_fwd = (cur > prev) & (cur >= 1)
        roll_back = (cur < prev) & (cur > 0) & (prev > 0)
        stagnant = (cur == prev) & (cur >= 2)
        # The latest transition uses a different code-1 convention (code 1 exists almost only
        # in PAY_1), so it is kept OUT of the 6-month counts: N_* / MAX_ROLL_FWD_STEP count the
        # 4 consistently coded history transitions (PAY_6 -> ... -> PAY_2); the latest one is
        # captured only by the *_RECENT flags. No double counting, one coding regime per feature.
        hist = slice(None, -1)
        step = np.where(roll_fwd, cur - np.maximum(prev, 0), 0)
        return {
            "CURED_RECENT": cure[:, -1],
            "N_CURES": cure[:, hist].sum(axis=1),
            "ROLL_FWD_RECENT": roll_fwd[:, -1],
            "N_ROLL_FWD": roll_fwd[:, hist].sum(axis=1),
            "MAX_ROLL_FWD_STEP": step[:, hist].max(axis=1),
            "N_DORMANT_TO_REVOLVER": ((prev == -2) & (cur == 0))[:, hist].sum(axis=1),
            "ROLL_BACK_RECENT": roll_back[:, -1],
            "N_ROLL_BACK": roll_back[:, hist].sum(axis=1),
            "STAGNANT_RECENT": stagnant[:, -1],
            "N_STAGNANT": stagnant[:, hist].sum(axis=1),
            "MAX_DELINQ_STREAK": _longest_run(chrono >= 2),
        }
