"""Tests for the API. The real model is replaced with a fake one,
so these tests run fast and do not download anything.
"""

import io
import os
import tempfile

# Use a throwaway database file for the tests.
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["TRIAGE_DB"] = _tmp.name

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import db, model
from app.main import app


@pytest.fixture(autouse=True)
def fresh_db():
    db.init_db()
    conn = db.connect()
    conn.execute("DELETE FROM cases")
    conn.commit()
    conn.close()
    yield


@pytest.fixture
def client():
    # Fake model: ready, and always says PNEUMONIA with 90 percent.
    model.state["status"] = "ready"
    model._model = None
    model.predict = lambda image: ("PNEUMONIA", 0.9)
    return TestClient(app)


def make_png():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "gray").save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


def test_predict_rejects_non_image(client):
    res = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert "not an image" in res.json()["detail"]


def test_predict_when_model_not_ready(client):
    model.state["status"] = "loading"
    res = client.post("/predict", files={"file": ("x.png", make_png(), "image/png")})
    assert res.status_code == 503
    model.state["status"] = "ready"


def test_predict_saves_case(client):
    res = client.post("/predict", files={"file": ("x.png", make_png(), "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "PNEUMONIA"
    assert body["probability"] == 0.9
    assert body["review_status"] == "pending"

    res = client.get("/cases")
    assert res.json()["total"] == 1


def test_predict_preprocessing_error(client, monkeypatch):
    def broken(image):
        raise ValueError("bad pixels")

    monkeypatch.setattr(model, "predict", broken)
    res = client.post("/predict", files={"file": ("x.png", make_png(), "image/png")})
    assert res.status_code == 422


def test_review_flow_and_stats(client):
    client.post("/predict", files={"file": ("a.png", make_png(), "image/png")})
    client.post("/predict", files={"file": ("b.png", make_png(), "image/png")})
    ids = [c["id"] for c in client.get("/cases").json()["cases"]]

    res = client.post(
        f"/cases/{ids[0]}/review", json={"decision": "confirmed", "note": "ok"}
    )
    assert res.status_code == 200
    assert res.json()["review_status"] == "confirmed"

    res = client.post(f"/cases/{ids[1]}/review", json={"decision": "overridden"})
    assert res.json()["review_status"] == "overridden"

    stats = client.get("/stats").json()
    assert stats["total"] == 2
    assert stats["confirmed"] == 1
    assert stats["overridden"] == 1
    assert stats["accuracy"] == 0.5


def test_review_missing_case(client):
    res = client.post("/cases/999/review", json={"decision": "confirmed"})
    assert res.status_code == 404


def test_review_bad_decision(client):
    client.post("/predict", files={"file": ("a.png", make_png(), "image/png")})
    case_id = client.get("/cases").json()["cases"][0]["id"]
    res = client.post(f"/cases/{case_id}/review", json={"decision": "maybe"})
    assert res.status_code == 422


def test_empty_case_list(client):
    res = client.get("/cases")
    assert res.status_code == 200
    assert res.json() == {"cases": [], "total": 0}
