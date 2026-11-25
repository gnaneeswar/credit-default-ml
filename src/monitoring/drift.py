"""Evidently data-drift + classification-quality monitoring.

Reference = training rows with their *out-of-fold* predictions (an honest baseline of how
the model performs on unseen data). Each scenario compares a current batch against it with
``DataDriftPreset`` (per-feature distribution tests) and ``ClassificationPreset`` (F1,
accuracy, precision, recall, ROC-AUC at the registered threshold).

Scenarios:
* ``holdout``         – the real 20% test set scored by @champion (sanity: no alerts expected)
* ``covariate_drift`` – simulated downturn: limits cut 40%, every repayment status one month
                        worse, bills +40%; re-scored by @champion (expect a data-drift alert)
* ``concept_drift``   – features unchanged, 30% of labels flipped: the feature->default
                        relationship changed (expect a performance alert *without* data drift
                        — why label-based monitoring is needed on top of drift detection)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging import get_logger, log_path_io
from src.data.schema import BILL_AMT_COLS, FEATURES, PAY_STATUS_COLS, TARGET

log = get_logger(__name__)

PROBA, PRED = "prediction_proba", "prediction"

# Alert policy (documented in README). Drift: share of the 23 features whose distribution
# distance exceeds Evidently's per-column threshold. Performance: absolute drop vs reference.
DRIFT_SHARE_ALERT = 0.25
F1_DROP_ALERT = 0.05
ACCURACY_DROP_ALERT = 0.05
CONCEPT_FLIP_RATE = 0.30

_CLASSIFICATION = {
    "F1Score": "f1",
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "RocAuc": "roc_auc",
}


# ------------------------------------------------------------------ scenarios
def simulate_covariate_drift(features: pd.DataFrame) -> pd.DataFrame:
    """Economic-downturn shift: lower limits, worse repayment status, higher balances."""
    out = features.copy()
    out["LIMIT_BAL"] = (out["LIMIT_BAL"] * 0.6).round()
    for col in PAY_STATUS_COLS:
        out[col] = (out[col] + 1).clip(upper=9)
    for col in BILL_AMT_COLS:
        out[col] = (out[col] * 1.4).round()
    return out


def simulate_concept_drift(y: pd.Series, seed: int = 42) -> pd.Series:
    """Flip a share of labels: same applicants, different outcomes."""
    rng = np.random.default_rng(seed)
    flip = rng.random(len(y)) < CONCEPT_FLIP_RATE
    return pd.Series(np.where(flip, 1 - y.to_numpy(), y.to_numpy()), index=y.index, name=y.name)


# ------------------------------------------------------------------ evidently
def _dataset(frame: pd.DataFrame):
    from evidently import BinaryClassification, DataDefinition, Dataset

    definition = DataDefinition(
        numerical_columns=FEATURES,
        classification=[
            BinaryClassification(
                target=TARGET, prediction_labels=PRED, prediction_probas=PROBA, pos_label=1
            )
        ],
    )
    return Dataset.from_pandas(frame[[*FEATURES, TARGET, PRED, PROBA]], data_definition=definition)


def _parse(snapshot_dict: dict) -> dict[str, Any]:
    drift: dict[str, Any] = {}
    columns: dict[str, dict] = {}
    quality: dict[str, float] = {}
    for m in snapshot_dict["metrics"]:
        cfg, value = m["config"], m["value"]
        kind = cfg["type"].rsplit(":", 1)[-1]
        if kind == "DriftedColumnsCount":
            drift = {"drifted_count": int(value["count"]), "drift_share": float(value["share"])}
        elif kind == "ValueDrift" and cfg.get("column") in FEATURES:
            score, threshold = float(value), float(cfg["threshold"])
            columns[cfg["column"]] = {
                "method": cfg["method"],
                "score": score,
                "threshold": threshold,
                "drifted": score >= threshold,  # distance methods: larger = more drift
            }
        elif kind in _CLASSIFICATION and isinstance(value, int | float):
            quality[_CLASSIFICATION[kind]] = float(value)
    drift["drifted_columns"] = sorted(c for c, v in columns.items() if v["drifted"])
    return {"drift": drift, "column_drift": columns, "classification": quality}


def run_report(reference: pd.DataFrame, current: pd.DataFrame, threshold: float):
    from evidently import Report
    from evidently.presets import ClassificationPreset, DataDriftPreset

    report = Report(
        [
            DataDriftPreset(columns=FEATURES, drift_share=DRIFT_SHARE_ALERT),
            ClassificationPreset(probas_threshold=threshold),
        ]
    )
    snapshot = report.run(_dataset(current), _dataset(reference))
    return snapshot, _parse(snapshot.dict())


def reference_quality(reference: pd.DataFrame) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y, p, s = reference[TARGET], reference[PRED], reference[PROBA]
    return {
        "f1": float(f1_score(y, p)),
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p)),
        "roc_auc": float(roc_auc_score(y, s)),
    }


def score_batch(
    features: pd.DataFrame, y: pd.Series, predict_proba: Callable, threshold: float
) -> pd.DataFrame:
    proba = predict_proba(features[FEATURES].astype("float64"))
    out = features[FEATURES].copy()
    out[TARGET] = y.to_numpy()
    out[PROBA] = proba
    out[PRED] = (proba >= threshold).astype(int)
    return out


def evaluate_scenario(
    name: str,
    description: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float,
    baseline: dict[str, float],
    html_path: Path | None = None,
) -> dict[str, Any]:
    """Run the Evidently report for one scenario; optionally save its HTML.

    Side-effect free apart from the HTML file and silent (no logging), so scenarios can run
    in parallel worker processes and the caller logs results in a deterministic order.
    """
    import warnings

    warnings.filterwarnings("ignore")
    snapshot, parsed = run_report(reference, current, threshold)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
    q = parsed["classification"]
    drift = parsed["drift"]
    f1_drop = baseline["f1"] - q.get("f1", np.nan)
    acc_drop = baseline["accuracy"] - q.get("accuracy", np.nan)
    return {
        "scenario": name,
        "description": description,
        "n_rows": int(len(current)),
        "flagged_share": float(current[PRED].mean()),
        **drift,
        "classification": q,
        "f1_drop": float(f1_drop),
        "accuracy_drop": float(acc_drop),
        "alerts": {
            "data_drift": drift["drift_share"] >= DRIFT_SHARE_ALERT,
            "f1_drop": f1_drop > F1_DROP_ALERT,
            "accuracy_drop": acc_drop > ACCURACY_DROP_ALERT,
        },
        "column_drift": parsed["column_drift"],
        "html": str(html_path) if html_path is not None else None,
    }


def log_scenario(summary: dict[str, Any]) -> None:
    q = summary["classification"]
    log.info(
        "%s: drift share %.2f (%d cols: %s) · F1 %.4f (Δ %+.4f) · accuracy %.4f (Δ %+.4f)",
        summary["scenario"],
        summary["drift_share"],
        summary["drifted_count"],
        ", ".join(summary["drifted_columns"]) or "none",
        q.get("f1", np.nan),
        -summary["f1_drop"],
        q.get("accuracy", np.nan),
        -summary["accuracy_drop"],
    )
    if summary["html"]:
        log_path_io(log, "write", Path(summary["html"]))
