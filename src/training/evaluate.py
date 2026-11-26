"""Holdout evaluation of the *registered* ``@champion`` at its registered threshold.

The model is re-loaded from the Model Registry (not reused from memory), via the same
``load_champion`` the API uses — so this also proves the registry round-trip works.
The 20% test set is used exactly once, here, after all tuning and threshold choices.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.config import Settings
from src.common.logging import get_logger, log_path_io
from src.common.mlflow_utils import load_champion
from src.common.tables import log_table
from src.training.threshold import banded_decile_table, population_stability_index

log = get_logger(__name__)


def classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "flagged_share": float(pred.mean()),
    }


def _confusion_png(cm: np.ndarray, threshold: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=120)
    ax.imshow(cm, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center", color=color)
    ax.set_xticks([0, 1], ["no default", "default"])
    ax.set_yticks([0, 1], ["no default", "default"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Holdout confusion matrix (threshold {threshold:.3f})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _native_importance(est, names: list[str]) -> tuple[np.ndarray, str]:
    """The model's own importance, in a comparable unit per family, normalised to sum 1.

    LightGBM's and XGBoost's default ``feature_importances_`` differ (LightGBM: *split
    counts*, which favour continuous features with many thresholds regardless of loss
    reduction), so both boosters are read explicitly as *total gain*.
    """
    kind = type(est).__name__
    if kind == "LGBMClassifier":
        raw, label = est.booster_.feature_importance(importance_type="gain"), "total gain"
    elif kind == "XGBClassifier":
        score = est.get_booster().get_score(importance_type="total_gain")
        raw, label = np.array([score.get(n, 0.0) for n in names]), "total gain"
    elif hasattr(est, "feature_importances_"):
        raw, label = est.feature_importances_, "impurity (MDI)"
    else:
        raw, label = np.abs(np.ravel(est.coef_)), "|coefficient|"
    raw = np.asarray(raw, dtype=float)
    return raw / raw.sum() if raw.sum() > 0 else raw, label


def feature_importance(model) -> pd.DataFrame:
    """Per-feature importance of the champion from the fitted model itself (no extra cost):
    total gain for LightGBM / XGBoost, impurity MDI for random forest, |coefficient| for LR.

    These are computed on training data and favour continuous / correlated features, so they
    are paired with the holdout *group* permutation importance below, which is the robust,
    model-agnostic measure.
    """
    from src.data.schema import RAW_FEATURE_GROUPS
    from src.features.domain import FEATURE_TO_GROUP

    raw_group = {f: g for g, feats, _ in RAW_FEATURE_GROUPS for f in feats}
    est = model[-1]
    names = [str(c) for c in model[:-1].get_feature_names_out()]
    native, native_type = _native_importance(est, names)
    base = [n.removeprefix("WOE_") for n in names]
    out = pd.DataFrame(
        {
            "feature": names,
            "origin": ["original" if b in raw_group else "engineered" for b in base],
            "group": [raw_group.get(b) or FEATURE_TO_GROUP.get(b, "other") for b in base],
            "importance": native,
            "importance_type": native_type,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def _group_drop(scorer, frame: pd.DataFrame, y: np.ndarray, cols: list[str], seed: int) -> float:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    shuffled = frame.copy()
    shuffled[cols] = shuffled[cols].to_numpy()[rng.permutation(len(shuffled))]  # shared perm
    return float(roc_auc_score(y, scorer.predict_proba(shuffled)[:, 1]))


def group_importance(
    model, X: pd.DataFrame, y: np.ndarray, seed: int = 42, n_repeats: int = 3
) -> pd.DataFrame:
    """Holdout permutation importance per feature group, as the mean drop in ROC-AUC.

    Two different questions, reported in the ``kind`` column:

    * ``total`` (original groups): the raw input columns are shuffled *before* the full
      pipeline, so every feature derived from them changes too. Answers "how much does the
      model rely on this source of information?" — e.g. shuffling PAY_1..6 also scrambles
      N_DELINQ, the roll-rate features and MAX_DELINQ_STREAK.
    * ``conditional`` (engineered groups): the group's columns are shuffled in the engineered
      matrix while everything else, including the raw columns they were derived from, stays
      intact. Answers "what does this group add on top of all other inputs?". Correlated
      inputs outside the group can stand in for it, so these values are lower bounds.

    Columns inside a group always share one row permutation.
    """
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score

    from src.data.schema import RAW_FEATURE_GROUPS
    from src.features.domain import FEATURE_GROUPS

    Xt = model[:-1].transform(X)
    est = model[-1]
    base = float(roc_auc_score(y, est.predict_proba(Xt)[:, 1]))
    present = {str(c).removeprefix("WOE_"): str(c) for c in Xt.columns}
    groups = []  # (kind, origin, name, scorer, frame, cols)
    for name, feats, _ in RAW_FEATURE_GROUPS:
        cols = [f for f in feats if f in X.columns]
        if cols:
            groups.append(("total", "original", name, model, X, cols))
    for name, feats, _ in FEATURE_GROUPS:
        cols = [present[f] for f in feats if f in present]
        if cols:
            groups.append(("conditional", "engineered", name, est, Xt, cols))
    jobs = [
        delayed(_group_drop)(scorer, frame, y, cols, seed + r)
        for _, _, _, scorer, frame, cols in groups
        for r in range(n_repeats)
    ]
    scores = np.array(Parallel(n_jobs=-1)(jobs)).reshape(len(groups), n_repeats)
    drops = base - scores
    out = pd.DataFrame(
        {
            "group": [g[2] for g in groups],
            "kind": [g[0] for g in groups],
            "origin": [g[1] for g in groups],
            "n_features": [len(g[5]) for g in groups],
            "auc_drop_mean": drops.mean(axis=1),
            "auc_drop_std": drops.std(axis=1),
        }
    ).sort_values(["kind", "auc_drop_mean"], ascending=[False, False], ignore_index=True)
    out.insert(0, "rank", out.groupby("kind").cumcount() + 1)
    return out


def evaluate_champion(
    X_test: pd.DataFrame, y_test: pd.Series, settings: Settings
) -> tuple[dict, str]:
    """Score the test set with @champion; write metrics + confusion matrix artifacts.

    Returns (metrics, champion run_id) so the caller can log artifacts to that run.
    """
    model, info = load_champion(settings)
    proba = model.predict_proba(X_test[info.input_columns].astype("float64"))[:, 1]
    y = y_test.to_numpy()

    at_threshold = classification_metrics(y, proba, info.threshold)
    at_default = classification_metrics(y, proba, 0.5)
    comparison = pd.DataFrame(
        [{"cutoff": "registered", **at_threshold}, {"cutoff": "default 0.5", **at_default}]
    )
    log_table(log, f"holdout metrics (n={len(y)}, champion v{info.version})", comparison)

    tn, fp, fn, tp = confusion_matrix(y, (proba >= info.threshold).astype(int)).ravel()
    cm = {
        "threshold": info.threshold,
        "labels": ["no_default", "default"],
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "missed_defaults": int(fn),
        "good_customers_flagged": int(fp),
    }
    log.info(
        "confusion @%.4f: TN=%d FP=%d FN=%d TP=%d (caught %d of %d defaults; flagged %d good customers)",
        info.threshold,
        tn,
        fp,
        fn,
        tp,
        tp,
        tp + fn,
        fp,
    )

    out = settings.artifacts_dir
    metrics = {
        "model": info.as_dict(),
        "n_test": int(len(y)),
        "test_default_rate": float(y.mean()),
        "at_registered_threshold": at_threshold,
        "at_default_0_5": at_default,
    }

    # Holdout deciles on the *training* OOF score bands: rows line up with the train table,
    # and the band shares measure score stability (PSI) on unseen data.
    train_deciles_path = out / "train_decile_analysis.csv"
    if train_deciles_path.exists():
        log_path_io(log, "read", train_deciles_path)
        train_deciles = pd.read_csv(train_deciles_path)
        test_deciles = banded_decile_table(y, proba, train_deciles)
        psi = population_stability_index(train_deciles["share"], test_deciles["share"])
        metrics["score_psi_vs_train_bands"] = psi
        log_table(log, "holdout deciles on the training score bands", test_deciles)
        log.info("score PSI (holdout vs train bands) = %.4f (< 0.1 = stable)", psi)
        path = out / "test_decile_analysis.csv"
        test_deciles.to_csv(path, index=False, lineterminator="\n")
        log_path_io(log, "write", path)
    else:
        log.warning("%s missing: skipping holdout decile table", train_deciles_path)

    fi = feature_importance(model)
    log_table(
        log,
        f"champion feature importance — {fi['importance_type'].iloc[0]} (top 15 of {len(fi)})",
        fi.head(15).drop(columns=["importance_type"]),
    )
    fi_path = out / "feature_importance.csv"
    fi.to_csv(fi_path, index=False, lineterminator="\n")
    log_path_io(log, "write", fi_path)
    t0 = time.perf_counter()
    gi = group_importance(model, X_test[info.input_columns].astype("float64"), y, settings.seed)
    log_table(log, f"group permutation importance on holdout ({time.perf_counter() - t0:.1f}s)", gi)
    gi_path = out / "feature_group_importance.csv"
    gi.to_csv(gi_path, index=False, lineterminator="\n")
    log_path_io(log, "write", gi_path)

    files = {
        out / "test_metrics.json": json.dumps(metrics, indent=2) + "\n",
        out / "test_confusion_matrix.json": json.dumps(cm, indent=2) + "\n",
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        log_path_io(log, "write", path)
    png = out / "test_confusion_matrix.png"
    _confusion_png(np.array(cm["matrix"]), info.threshold, png)
    log_path_io(log, "write", png)
    return metrics, info.run_id
