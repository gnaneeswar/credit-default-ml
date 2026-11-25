"""Out-of-fold scoring and decile-based F1 threshold selection.

``cross_val_predict`` with the same 3-fold splitter gives every training row exactly one
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

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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


def oof_probabilities(
    estimator: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> np.ndarray:
    proba = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=-1)
    return proba[:, 1]


def decile_table(y: pd.Series | np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """10-row lift table; row k also reports metrics for the cutoff 'flag deciles 1..k'."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(-proba, kind="stable")
    decile_of_rank = np.arange(len(proba)) * 10 // len(proba) + 1  # rank-based: no ties issue
    decile = np.empty(len(proba), dtype=int)
    decile[order] = decile_of_rank
    df = pd.DataFrame({"decile": decile, "y": y, "p": proba})
    base_rate = y.mean()
    total_bad = y.sum()

    g = df.groupby("decile").agg(
        n=("y", "size"), n_default=("y", "sum"), min_score=("p", "min"), max_score=("p", "max")
    )
    g["bad_rate"] = g["n_default"] / g["n"]
    g["lift"] = g["bad_rate"] / base_rate
    g["cum_n"] = g["n"].cumsum()
    g["cum_default"] = g["n_default"].cumsum()
    g["cum_capture"] = g["cum_default"] / total_bad
    g["cum_lift"] = (g["cum_default"] / g["cum_n"]) / base_rate

    # Metrics if the cutoff is this decile's lower bound (score >= min_score is flagged).
    rows = []
    for k, cutoff in g["min_score"].items():
        f1, precision, recall = _metrics_at(y, proba, cutoff)
        rows.append(
            {"decile": k, "cutoff": cutoff, "precision": precision, "recall": recall, "f1": f1}
        )
    cut = pd.DataFrame(rows).set_index("decile")
    out = g.join(cut).reset_index()
    return out[
        [
            "decile",
            "n",
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
    ]


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
