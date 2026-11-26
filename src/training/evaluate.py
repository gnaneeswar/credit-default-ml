"""Holdout evaluation of the *registered* ``@champion`` at its registered threshold.

The model is re-loaded from the Model Registry (not reused from memory), via the same
``load_champion`` the API uses — so this also proves the registry round-trip works.
The 20% test set is used exactly once, here, after all tuning and threshold choices.
"""

from __future__ import annotations

import json
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
