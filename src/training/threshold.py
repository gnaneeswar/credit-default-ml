"""Out-of-fold scoring and decile-based F1 threshold selection.

Out-of-fold scoring with the same 3-fold splitter gives every training row exactly one
prediction from a model that never saw it — an honest, full-coverage basis for choosing a
cutoff without touching the test set or carving out a static validation split.

Coarse-to-fine, policy-shaped search:

1. **Deciles** rank customers by predicted risk (decile 1 = riskiest 10%). The decile table
   (bad rate, lift, capture, and F1 if we "flag the top k deciles") picks the best k.
2. **Percentile refinement** searches "flag the top q%" cutoffs (q in 1% steps) only within
   the deciles adjacent to k. Each step moves ~1% of customers, so the optimum cannot chase
   individual noisy scores, yet it is not stuck on 10%-wide steps that can land below the
   default 0.5 cutoff.

The cutoff is therefore always expressible as a credit policy ("review the riskiest 27%").
The unconstrained fine-grained F1 optimum and the default 0.5 are reported for comparison.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.common.logging import get_logger
from src.common.tables import log_table

log = get_logger(__name__)


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    deciles_flagged: int  # coarse step: best "flag top k deciles"
    decile_f1: float
    top_percent_flagged: int  # fine step: best "flag top q%" within deciles k-1..k+1
    f1: float
    precision: float
    recall: float
    flagged_share: float
    fine_threshold: float
    fine_f1: float
    f1_at_050: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _fit(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    return estimator.fit(X, y)


def refit_with_oof(
    template: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    extra: dict[str, Pipeline] | None = None,
) -> tuple[Pipeline, np.ndarray, dict[str, Pipeline]]:
    """One parallel batch: full-train refit of the winner, its k out-of-fold fits, and any
    ``extra`` full-train refits (e.g. LR for the IV table).

    Equivalent to ``refit`` + ``cross_val_predict(method="predict_proba")`` with the same
    splitter, but the k+1+len(extra) independent fits run concurrently instead of serially.
    Returns (refitted winner, OOF P(default) for every training row, refitted extras).
    """
    folds = list(cv.split(X, y))
    extra = extra or {}
    # The full-data refit is the longest job in this batch; give it the cores the other
    # fits leave idle, then restore n_jobs=1 so the logged/served model is single-threaded.
    spare = max(1, (os.cpu_count() or 1) - len(folds) - len(extra))
    full = clone(template)
    threaded = full.get_params().get("model__n_jobs") == 1  # tree models we pinned explicitly
    if threaded:
        full.set_params(model__n_jobs=spare)
    jobs = [delayed(_fit)(full, X, y)]
    jobs += [delayed(_fit)(clone(template), X.iloc[tr], y.iloc[tr]) for tr, _ in folds]
    jobs += [delayed(_fit)(clone(est), X, y) for est in extra.values()]
    fitted = Parallel(n_jobs=-1)(jobs)

    winner, fold_models, extra_fitted = (
        fitted[0],
        fitted[1 : 1 + len(folds)],
        fitted[1 + len(folds) :],
    )
    oof = np.empty(len(X), dtype=float)
    for model, (_, te) in zip(fold_models, folds, strict=True):
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return winner, oof, dict(zip(extra, extra_fitted, strict=True))


DECILE_COLUMNS = [
    "decile",
    "n",
    "share",
    "n_default",
    "bad_rate",
    "lift",
    "cum_capture",
    "cum_lift",
    "min_score",
    "max_score",
    "cutoff",
    "precision",
    "recall",
    "f1",
]


def _summarise_bands(
    y: np.ndarray, proba: np.ndarray, band: np.ndarray, cutoffs: dict[int, float]
) -> pd.DataFrame:
    """Per-band lift table. ``cutoffs[k]`` is band k's lower score bound: row k also reports
    the metrics of the policy 'flag every account with score >= cutoffs[k]'. Empty bands
    (possible on new data) are kept as zero rows rather than dropped."""
    df = pd.DataFrame({"decile": band, "y": y, "p": proba})
    base_rate, total_bad = y.mean(), y.sum()
    g = (
        df.groupby("decile")
        .agg(
            n=("y", "size"), n_default=("y", "sum"), min_score=("p", "min"), max_score=("p", "max")
        )
        .reindex(range(1, 11))
    )
    g[["n", "n_default"]] = g[["n", "n_default"]].fillna(0).astype(int)
    g["share"] = g["n"] / len(y)
    g["bad_rate"] = np.divide(g["n_default"], g["n"], out=np.zeros(10), where=g["n"] > 0)
    g["lift"] = g["bad_rate"] / base_rate
    cum_n, cum_bad = g["n"].cumsum(), g["n_default"].cumsum()
    g["cum_capture"] = cum_bad / total_bad
    g["cum_lift"] = np.divide(cum_bad, cum_n, out=np.zeros(10), where=cum_n > 0) / base_rate
    g["cutoff"] = pd.Series(cutoffs)
    metrics = [_metrics_at(y, proba, c) for c in g["cutoff"]]
    g["f1"], g["precision"], g["recall"] = zip(*metrics, strict=True)
    return g.reset_index()[DECILE_COLUMNS]


def decile_table(y: pd.Series | np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """10-row lift table on equal-count score deciles (decile 1 = riskiest 10%)."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(-proba, kind="stable")
    decile = np.empty(len(proba), dtype=int)
    decile[order] = np.arange(len(proba)) * 10 // len(proba) + 1  # rank-based: no ties issue
    cutoffs = pd.Series(proba).groupby(decile).min().to_dict()
    return _summarise_bands(y, proba, decile, cutoffs)


def banded_decile_table(
    y: pd.Series | np.ndarray, proba: np.ndarray, reference: pd.DataFrame
) -> pd.DataFrame:
    """Lift table for a *new* batch using the reference (training OOF) decile score bands.

    Band k = scores in [reference cutoff_k, cutoff_(k-1)); scores below the reference's
    lowest cutoff fall into band 10. Rows therefore line up with the training table, and the
    per-band ``share`` shows whether the score distribution is stable (ideal: 10% each).
    """
    y = np.asarray(y, dtype=int)
    cut = reference.set_index("decile")["cutoff"].sort_index()
    band = 1 + (np.asarray(proba)[:, None] < cut.loc[1:9].to_numpy()[None, :]).sum(axis=1)
    return _summarise_bands(y, np.asarray(proba), band, cut.to_dict())


def population_stability_index(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    """PSI between two band-share distributions (< 0.1 stable, 0.1-0.25 shift, > 0.25 major)."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(((a - e) * np.log(a / e)).sum())


def _metrics_at(y: np.ndarray, proba: np.ndarray, cutoff: float) -> tuple[float, float, float]:
    pred = proba >= cutoff
    tp = int((pred & (y == 1)).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int(y.sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def choose_threshold(
    y: pd.Series | np.ndarray, proba: np.ndarray, table: pd.DataFrame
) -> ThresholdChoice:
    y = np.asarray(y, dtype=int)
    best_decile = table.loc[table["f1"].idxmax()]
    k = int(best_decile["decile"])

    # Percentile refinement inside deciles k-1..k+1 ("flag the top q%").
    q_lo, q_hi = max(1, 10 * (k - 1)), min(100, 10 * (k + 1))
    candidates = []
    for q in range(q_lo, q_hi + 1):
        cutoff = float(np.quantile(proba, 1 - q / 100))
        f1, precision, recall = _metrics_at(y, proba, cutoff)
        candidates.append((f1, q, cutoff, precision, recall))
    f1, q, cutoff, precision, recall = max(candidates)

    prec, rec, thr = precision_recall_curve(y, proba)
    f1s = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)[:-1]
    j = int(np.argmax(f1s))
    choice = ThresholdChoice(
        threshold=cutoff,
        deciles_flagged=k,
        decile_f1=float(best_decile["f1"]),
        top_percent_flagged=q,
        f1=f1,
        precision=precision,
        recall=recall,
        flagged_share=float((proba >= cutoff).mean()),
        fine_threshold=float(thr[j]),
        fine_f1=float(f1s[j]),
        f1_at_050=float(f1_score(y, proba >= 0.5)),
    )
    log_table(log, "OOF decile analysis (decile 1 = riskiest)", table)
    log.info(
        "coarse: best = flag top %d deciles (cutoff %.4f, F1=%.4f)",
        k,
        float(best_decile["cutoff"]),
        choice.decile_f1,
    )
    log.info(
        "fine: flag top %d%% -> cutoff=%.4f  F1=%.4f  P=%.4f  R=%.4f  (searched top %d%%..%d%%)",
        q,
        cutoff,
        f1,
        precision,
        recall,
        q_lo,
        q_hi,
    )
    log.info(
        "reference: unconstrained optimum %.4f F1=%.4f · default 0.5 F1=%.4f",
        choice.fine_threshold,
        choice.fine_f1,
        choice.f1_at_050,
    )
    return choice
