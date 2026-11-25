"""Batch-score the held-out test set against the running API in one HTTP request.

    uv run python -m src.inference.batch_client [--url http://localhost:8000]

Reads ``artifacts/holdout/test.parquet`` (written by training), drops the target column
before building the ``dataframe_split`` payload (the API rejects unknown columns), keeps the
labels locally, and writes ``artifacts/batch_predictions.csv`` with actual vs predicted.
Standalone by design: no imports from training or data code.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from src.common.config import get_settings
from src.common.logging import get_logger, log_path_io, stage
from src.common.tables import log_table

log = get_logger("inference.batch_client")

DEFAULT_TARGET = "default_payment_next_month"


def build_payload(features: pd.DataFrame) -> dict:
    split = features.to_dict(orient="split")
    return {
        "dataframe_split": {
            "columns": split["columns"],
            "data": split["data"],
            "index": [int(i) for i in split["index"]],
        }
    }


def score(
    url: str,
    input_path: Path,
    output_path: Path,
    target: str = DEFAULT_TARGET,
    timeout: float = 120.0,
) -> pd.DataFrame:
    log_path_io(log, "read", input_path)
    df = pd.read_parquet(input_path)
    actual = df.pop(target) if target in df.columns else None
    log.info(
        "payload: %d rows x %d feature columns (target '%s' %s)",
        len(df),
        df.shape[1],
        target,
        "dropped" if actual is not None else "not present",
    )

    payload = build_payload(df)
    endpoint = f"{url.rstrip('/')}/invocations"
    start = time.perf_counter()
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    if resp.status_code != 200:
        raise RuntimeError(f"POST {endpoint} -> {resp.status_code}: {resp.text[:1000]}")
    body = resp.json()
    log.info(
        "POST %s -> 200 in %.2fs (%d rows, model %s v%s, threshold %.4f)",
        endpoint,
        elapsed,
        body["n_rows"],
        body["model_name"],
        body["model_version"],
        body["threshold"],
    )

    out = pd.DataFrame(body["predictions"], index=body.get("index") or df.index)
    out.index.name = "row_id"
    if actual is not None:
        out.insert(0, "actual", actual.to_numpy())
    out = out.reset_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, lineterminator="\n")
    log_path_io(log, "write", output_path)

    summary = {"rows": len(out), "flagged_share": float(out["prediction"].mean())}
    if actual is not None:
        from sklearn.metrics import f1_score, precision_score, recall_score

        y, p = out["actual"], out["prediction"]
        summary |= {
            "f1": f1_score(y, p),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p),
        }
    log_table(log, "batch scoring summary", pd.DataFrame([summary]))
    return out


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=s.api_url)
    parser.add_argument("--input", type=Path, default=s.holdout_dir / "test.parquet")
    parser.add_argument("--output", type=Path, default=s.artifacts_dir / "batch_predictions.csv")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    with stage("batch_score"):
        score(args.url, args.input, args.output, args.target, args.timeout)


if __name__ == "__main__":
    main()
