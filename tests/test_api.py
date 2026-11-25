"""Phase 4: FastAPI /health and /invocations contract, batch client, sample request."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.common.mlflow_utils import ChampionInfo
from src.data.schema import FEATURES, TARGET
from src.inference import batch_client
from src.inference.app import create_app
from tests.conftest import REPO_ROOT, TINY_THRESHOLD

THRESHOLD = 0.5


class FakeModel:
    """Deterministic stand-in: asserts the app hands over canonical order and float64."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert list(X.columns) == FEATURES, "app must reorder columns to the signature"
        assert all(dt == np.float64 for dt in X.dtypes), "app must cast to float64"
        p = np.clip((X["PAY_0"].to_numpy() + 2) / 10, 0, 1)
        return np.column_stack([1 - p, p])


INFO = ChampionInfo(
    name="credit-default-pipeline",
    alias="champion",
    version="7",
    run_id="abc123",
    threshold=THRESHOLD,
    input_columns=FEATURES,
    input_types=dict.fromkeys(FEATURES, "double"),
    tags={"model_family": "fake"},
)


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(loader=lambda: (FakeModel(), INFO))) as c:
        yield c


def _row(pay_0: int = 0) -> list[int]:
    row = dict.fromkeys(FEATURES, 1000)
    row.update(SEX=1, EDUCATION=2, MARRIAGE=1, AGE=30, PAY_0=pay_0)
    return [row[c] for c in FEATURES]


def _payload(rows: list[list], columns: list[str] = FEATURES, **extra) -> dict:
    return {"dataframe_split": {"columns": columns, "data": rows, **extra}}


# ---------------------------------------------------------------------- health
def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == "7" and body["threshold"] == THRESHOLD
    assert body["n_features"] == 23


def test_health_and_invocations_503_when_model_missing() -> None:
    def broken():
        raise RuntimeError("Registered model alias champion not found")

    with TestClient(create_app(loader=broken)) as c:
        r = c.get("/health")
        assert r.status_code == 503
        assert r.json()["model_loaded"] is False and "champion" in r.json()["error"]
        assert c.post("/invocations", json=_payload([_row()])).status_code == 503


# ------------------------------------------------------------------ happy path
def test_invocations_scores_and_thresholds(client: TestClient) -> None:
    r = client.post("/invocations", json=_payload([_row(-2), _row(3), _row(8)], index=[10, 11, 12]))
    assert r.status_code == 200
    body = r.json()
    probs = [p["default_probability"] for p in body["predictions"]]
    assert probs == pytest.approx([0.0, 0.5, 1.0])
    assert [p["prediction"] for p in body["predictions"]] == [0, 1, 1]  # >= threshold
    assert body["index"] == [10, 11, 12] and body["n_rows"] == 3
    assert body["threshold"] == THRESHOLD and body["model_version"] == "7"


def test_invocations_accepts_any_column_order(client: TestClient) -> None:
    shuffled = list(reversed(FEATURES))
    row = dict(zip(FEATURES, _row(3), strict=True))
    r = client.post("/invocations", json=_payload([[row[c] for c in shuffled]], shuffled))
    assert r.status_code == 200
    assert r.json()["predictions"][0]["default_probability"] == pytest.approx(0.5)


def test_invocations_accepts_floats(client: TestClient) -> None:
    r = client.post("/invocations", json=_payload([[float(v) for v in _row(1)]]))
    assert r.status_code == 200


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        (_payload([_row()[:-1]], FEATURES[:-1]), "missing_columns"),
        (_payload([[*_row(), 1]], [*FEATURES, "EXTRA"]), "unknown_columns"),
        (_payload([_row()[:-1]]), "row 0 has 22 values"),
        (_payload([["abc", *_row()[1:]]]), "int"),
        (_payload([[True, *_row()[1:]]]), "int"),
        (_payload([[None, *_row()[1:]]]), "int"),
        (_payload([]), "at least 1"),
        (_payload([[*_row(), 1]], [*FEATURES[:-1], "PAY_0", "PAY_0"]), "duplicate columns"),
        (_payload([_row()], index=[1, 2]), "index has 2 ids"),
        ({"columns": FEATURES, "data": [_row()]}, "dataframe_split"),
        ({**_payload([_row()]), "extra": 1}, "extra"),
    ],
    ids=[
        "missing-col",
        "unknown-col",
        "short-row",
        "string",
        "bool",
        "null",
        "empty",
        "duplicate-col",
        "index-len",
        "no-wrapper",
        "extra-key",
    ],
)
def test_invocations_rejects_bad_payloads(client: TestClient, payload: dict, fragment: str) -> None:
    r = client.post("/invocations", json=payload)
    assert r.status_code == 422, r.text
    assert fragment in r.text


def test_invocations_rejects_nan(client: TestClient) -> None:
    body = json.dumps(_payload([_row()])).replace("1000", "NaN", 1)
    r = client.post("/invocations", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_openapi_documents_both_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/invocations" in paths and "/health" in paths


# ------------------------------------------------------------ sample request
def test_sample_request_is_valid(client: TestClient) -> None:
    payload = json.loads((REPO_ROOT / "sample_request.json").read_text(encoding="utf-8"))
    assert payload["dataframe_split"]["columns"] == FEATURES
    r = client.post("/invocations", json=payload)
    assert r.status_code == 200 and r.json()["n_rows"] == 3


# -------------------------------------------------------------- batch client
def test_batch_client_drops_target_and_writes_csv(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    df = pd.DataFrame([_row(-2), _row(5), _row(0)], columns=FEATURES)
    df[TARGET] = [0, 1, 0]
    src = tmp_path / "test.parquet"
    df.to_parquet(src, index=False)
    sent = {}

    def fake_post(url, json, timeout):  # route the HTTP call into the TestClient
        sent.update(json)
        return client.post("/invocations", json=json)

    monkeypatch.setattr(batch_client.requests, "post", fake_post)
    out_path = tmp_path / "preds.csv"
    out = batch_client.score("http://testserver", src, out_path)

    assert TARGET not in sent["dataframe_split"]["columns"]
    assert len(sent["dataframe_split"]["data"]) == 3  # one request for all rows
    saved = pd.read_csv(out_path)
    assert list(saved.columns) == ["row_id", "actual", "default_probability", "prediction"]
    assert saved["actual"].tolist() == [0, 1, 0] and saved["prediction"].tolist() == [0, 1, 0]
    assert len(out) == 3


# ------------------------------------------------- integration: real registry
def test_app_serves_real_registered_champion(small_xy, tiny_champion) -> None:
    """Default loader: models:/credit-default-pipeline@champion from the SQLite registry."""
    X, _ = small_xy
    pipe, version, _ = tiny_champion
    rows = X.head(5)
    with TestClient(create_app()) as c:
        health = c.get("/health").json()
        assert health["model_version"] == version
        assert health["threshold"] == pytest.approx(TINY_THRESHOLD)
        r = c.post(
            "/invocations",
            json=_payload(rows.astype(int).values.tolist(), list(rows.columns)),
        )
    assert r.status_code == 200
    got = [p["default_probability"] for p in r.json()["predictions"]]
    np.testing.assert_allclose(got, pipe.predict_proba(rows)[:, 1], rtol=1e-9)
