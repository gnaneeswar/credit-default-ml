"""Weight-of-Evidence encoding with Information-Value feature selection.

``WoEIVSelector`` learns bins, WoE values and IV scores **only from the data passed to
``fit``**. Inside a Pipeline evaluated by GridSearchCV / cross_val_predict it is therefore
re-fitted on each training fold and applied to the held-out fold with frozen parameters —
the held-out labels never influence the encoding.

Binning (all features are numeric/ordinal in this dataset):
* columns with <= ``max_discrete`` distinct values (PAY_n codes, SEX, EDUCATION, MARRIAGE,
  month counts) start with one bin per value; others start with up to ``max_bins`` quantile
  bins;
* bins holding < ``min_bin_frac`` of rows are merged into their smaller adjacent neighbour,
  so each WoE estimate rests on enough rows (e.g. PAY_0 delays 4..8 merge into "3+").

WoE_i = ln(%good_i / %bad_i) with additive (Laplace) smoothing ``alpha``; higher = safer.
IV = sum_i (%good_i - %bad_i) * WoE_i. Features with IV < ``iv_min`` are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

IV_STRENGTH = [
    (0.02, "useless"),
    (0.10, "weak"),
    (0.30, "medium"),
    (0.50, "strong"),
    (np.inf, "very strong (verify)"),
]


def iv_strength(iv: float) -> str:
    return next(label for bound, label in IV_STRENGTH if iv < bound)


def _initial_cuts(x: np.ndarray, max_bins: int, max_discrete: int) -> np.ndarray:
    """Cut points c_1 < … < c_k; bin index = number of cuts strictly below the value."""
    uniq = np.unique(x)
    if len(uniq) <= 1:
        return np.array([], dtype=float)
    if len(uniq) <= max_discrete:
        return (uniq[:-1] + uniq[1:]) / 2.0  # one bin per distinct value
    cuts = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1]))
    # A cut equal to the max would leave an empty top bin.
    return cuts[cuts < uniq[-1]]


def _assign(x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.searchsorted(cuts, x, side="left")


def _merge_small_bins(x: np.ndarray, cuts: np.ndarray, min_count: float) -> np.ndarray:
    cuts = cuts.copy()
    while len(cuts) > 0:
        counts = np.bincount(_assign(x, cuts), minlength=len(cuts) + 1)
        i = int(np.argmin(counts))
        if counts[i] >= min_count:
            break
        if i == 0:
            drop = 0  # merge with right neighbour
        elif i == len(counts) - 1:
            drop = i - 1  # merge with left neighbour
        else:
            drop = i - 1 if counts[i - 1] <= counts[i + 1] else i
        cuts = np.delete(cuts, drop)
    return cuts


def woe_iv(bins: np.ndarray, y: np.ndarray, n_bins: int, alpha: float) -> tuple[np.ndarray, float]:
    """Smoothed WoE per bin and total IV from bin indices and a binary target."""
    bad = np.bincount(bins, weights=y, minlength=n_bins)
    total = np.bincount(bins, minlength=n_bins).astype(float)
    good = total - bad
    dist_bad = (bad + alpha) / (bad.sum() + alpha * n_bins)
    dist_good = (good + alpha) / (good.sum() + alpha * n_bins)
    woe = np.log(dist_good / dist_bad)
    iv = float(((dist_good - dist_bad) * woe).sum())
    return woe, iv


class WoEIVSelector(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        max_bins: int = 10,
        max_discrete: int = 12,
        min_bin_frac: float = 0.01,
        alpha: float = 0.5,
        iv_min: float = 0.02,
    ):
        self.max_bins = max_bins
        self.max_discrete = max_discrete
        self.min_bin_frac = min_bin_frac
        self.alpha = alpha
        self.iv_min = iv_min

    def fit(self, X: pd.DataFrame, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WoEIVSelector expects a pandas DataFrame")
        y_arr = np.asarray(y, dtype=float)
        if not np.isin(y_arr, (0.0, 1.0)).all():
            raise ValueError("WoEIVSelector requires a binary 0/1 target")
        if y_arr.min() == y_arr.max():
            raise ValueError("WoEIVSelector requires both classes in the fitting data")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        min_count = self.min_bin_frac * len(X)
        self.cuts_: dict[str, np.ndarray] = {}
        self.woe_: dict[str, np.ndarray] = {}
        rows = []
        for col in X.columns:
            x = X[col].to_numpy(dtype=float)
            observed = ~np.isnan(x)
            xo, yo = x[observed], y_arr[observed]
            cuts = _merge_small_bins(
                xo, _initial_cuts(xo, self.max_bins, self.max_discrete), min_count
            )
            woe, iv = woe_iv(_assign(xo, cuts), yo, len(cuts) + 1, self.alpha)
            self.cuts_[col] = cuts
            self.woe_[col] = woe
            rows.append({"feature": col, "iv": iv, "n_bins": len(cuts) + 1})

        table = pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)
        table["strength"] = table["iv"].map(iv_strength)
        table["selected"] = table["iv"] >= self.iv_min
        # Stored as plain records (not a DataFrame) so the fitted pipeline serialises with
        # skops, which cannot persist pandas internals. ``iv_table_`` rebuilds the frame.
        self.iv_records_ = [
            {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
            for rec in table.to_dict(orient="records")
        ]
        self.selected_features_ = table.loc[table["selected"], "feature"].tolist()
        if not self.selected_features_:
            # Never hand an empty matrix to the estimator: keep the single best feature.
            self.selected_features_ = [table.loc[0, "feature"]]
        return self

    @property
    def iv_table_(self) -> pd.DataFrame:
        """IV ranking: feature, iv, n_bins, strength, selected (sorted by IV desc)."""
        return pd.DataFrame(self.iv_records_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "iv_records_")
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(f"WoEIVSelector: missing columns {missing}")
        out = {}
        for col in self.selected_features_:
            x = X[col].to_numpy(dtype=float)
            woe = self.woe_[col][_assign(np.nan_to_num(x, nan=0.0), self.cuts_[col])]
            # Values outside the training range fall into the edge bins via searchsorted;
            # missing values carry no evidence -> WoE 0.
            out[f"WOE_{col}"] = np.where(np.isnan(x), 0.0, woe)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "iv_records_")
        return np.asarray([f"WOE_{c}" for c in self.selected_features_], dtype=object)

    def bin_table(self, col: str) -> pd.DataFrame:
        """Human-readable bins for one feature (for reports / interview walkthroughs)."""
        check_is_fitted(self, "iv_records_")
        cuts = self.cuts_[col]
        lows = np.concatenate([[-np.inf], cuts])
        highs = np.concatenate([cuts, [np.inf]])
        return pd.DataFrame({"low_excl": lows, "high_incl": highs, "woe": self.woe_[col]})
