"""Tests for the FastAPI reference inference server (server.py).

These require ``anno-sdk[server]`` (fastapi + uvicorn) to be installed.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from anno_sdk import (
    Annotation,
    Box2D,
    InferenceRequestMeta,
    InferenceResponse,
    InferenceServer,
    Polygon2D,
    Predictor,
    create_app,
)
from anno_sdk.server import _import_predictor


# ---------------------------------------------------------------------------
# Predictor stubs
# ---------------------------------------------------------------------------


class _EchoPredictor(Predictor):
    """Returns a single box inscribed in the image dimensions from metadata."""

    def predict(self, image_bytes, meta: InferenceRequestMeta) -> list[Annotation]:
        w = meta.width or 100
        h = meta.height or 100
        return [Annotation.from_geometry(Box2D(0, 0, float(w), float(h)), label=1)]


class _FailingPredictor(Predictor):
    def predict(self, image_bytes, meta):
        raise RuntimeError("model crashed")


class _FailingSetupPredictor(Predictor):
    def setup(self):
        raise RuntimeError("weights not found")


# ---------------------------------------------------------------------------
# Fixture: TestClient wrapping an InferenceServer app
# ---------------------------------------------------------------------------


def _make_client(predictor=None, **server_kwargs):
    srv = InferenceServer(predictor or _EchoPredictor(), host="127.0.0.1", port=8000, **server_kwargs)
    return TestClient(srv._app)


# ---------------------------------------------------------------------------
# Health / ready
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health(self) -> None:
        c = _make_client()
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_ready(self) -> None:
        c = _make_client()
        r = c.get("/ready")
        assert r.status_code == 200
        assert r.json() == {"status": "ready"}

    def test_ready_setup_failure(self) -> None:
        class _InitFail(Predictor):
            _called = False

            def setup(self):
                self._called = True
                raise RuntimeError("weights missing")

            def predict(self, image_bytes, meta):
                return []

        c = _make_client(_InitFail())
        r = c.get("/ready")
        assert r.status_code == 503
        assert "weights missing" in r.json()["error"]


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_annotations(self) -> None:
        c = _make_client()
        meta = {"image_id": 1, "task_id": 2, "label_mapping": {}, "requested_types": ["box"]}
        r = c.post(
            "/predict",
            files={"image": ("a.png", b"\x89PNG", "image/png")},
            data={"metadata": json.dumps(meta)},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["annotations"]) == 1
        assert body["annotations"][0]["annotation_type"] == "box"

    def test_response_is_re_parseable(self) -> None:
        c = _make_client()
        meta = {"image_id": 1, "task_id": 2, "label_mapping": {}, "requested_types": ["box"]}
        r = c.post(
            "/predict",
            files={"image": ("a.png", b"\x89PNG", "image/png")},
            data={"metadata": json.dumps(meta)},
        )
        parsed = InferenceResponse.from_dict(r.json())
        assert len(parsed.annotations) == 1
        assert isinstance(parsed.annotations[0].geometry, Box2D)

    def test_image_bytes_passed_to_predictor(self) -> None:
        captured: dict = {}

        class _Capture(Predictor):
            def predict(self, image_bytes, meta):
                captured["bytes"] = image_bytes
                return []

        c = _make_client(_Capture())
        meta = {"image_id": 1, "task_id": 2, "label_mapping": {}, "requested_types": []}
        c.post(
            "/predict",
            files={"image": ("a.png", b"\x89PNG-bytes", "image/png")},
            data={"metadata": json.dumps(meta)},
        )
        assert captured["bytes"] == b"\x89PNG-bytes"

    def test_predict_reads_image_dimensions_from_meta(self) -> None:
        captured: dict = {}

        class _DimCapture(Predictor):
            def predict(self, image_bytes, meta):
                captured["meta"] = meta
                return []

        c = _make_client(_DimCapture())
        meta = {"image_id": 1, "task_id": 2, "label_mapping": {"a": 0}, "requested_types": ["box"]}
        c.post(
            "/predict",
            files={"image": ("a.png", b"", "image/png")},
            data={"metadata": json.dumps(meta)},
        )
        assert captured["meta"].image_id == 1
        assert captured["meta"].label_mapping == {"a": 0}

    def test_missing_metadata_returns_400(self) -> None:
        c = _make_client()
        r = c.post("/predict", files={"image": ("a.png", b"", "image/png")})
        assert r.status_code == 422  # FastAPI validation

    def test_bad_metadata_json_returns_400(self) -> None:
        c = _make_client()
        r = c.post(
            "/predict",
            files={"image": ("a.png", b"", "image/png")},
            data={"metadata": "not-json"},
        )
        assert r.status_code == 400
        assert "invalid metadata JSON" in r.json()["detail"]

    def test_predict_failure_returns_500(self) -> None:
        c = _make_client(_FailingPredictor())
        meta = {"image_id": 1, "task_id": 2, "label_mapping": {}, "requested_types": ["box"]}
        r = c.post(
            "/predict",
            files={"image": ("a.png", b"x", "image/png")},
            data={"metadata": json.dumps(meta)},
        )
        assert r.status_code == 500
        assert "model crashed" in r.json()["detail"]

    def test_swagger_docs_accessible(self) -> None:
        c = _make_client()
        r = c.get("/docs")
        assert r.status_code == 200

    def test_redoc_accessible(self) -> None:
        c = _make_client()
        r = c.get("/redoc")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_predict_without_auth_returns_401(self) -> None:
        c = _make_client(auth_header="X-API-Key", auth_header_value="s3cr3t")
        r = c.post("/predict", files={"image": ("a.png", b"", "image/png")})
        assert r.status_code == 401

    def test_predict_with_correct_auth_succeeds(self) -> None:
        c = _make_client(auth_header="X-API-Key", auth_header_value="s3cr3t")
        meta = {"image_id": 1, "task_id": 2, "label_mapping": {}, "requested_types": ["box"]}
        r = c.post(
            "/predict",
            files={"image": ("a.png", b"\x89PNG", "image/png")},
            data={"metadata": json.dumps(meta)},
            headers={"X-API-Key": "s3cr3t"},
        )
        assert r.status_code == 200

    def test_predict_with_wrong_auth_returns_401(self) -> None:
        c = _make_client(auth_header="X-API-Key", auth_header_value="s3cr3t")
        r = c.post(
            "/predict",
            files={"image": ("a.png", b"", "image/png")},
            headers={"X-API-Key": "wrong"},
        )
        assert r.status_code == 401

    def test_query_auth(self) -> None:
        c = _make_client(auth_query="token", auth_query_value="tok123")
        meta = {"image_id": 1, "task_id": 2, "label_mapping": {}, "requested_types": ["box"]}

        # correct
        r = c.post(
            "/predict?token=tok123",
            files={"image": ("a.png", b"\x89PNG", "image/png")},
            data={"metadata": json.dumps(meta)},
        )
        assert r.status_code == 200

        # wrong
        r2 = c.post(
            "/predict?token=bad",
            files={"image": ("a.png", b"", "image/png")},
        )
        assert r2.status_code == 401

    def test_health_bypasses_auth(self) -> None:
        c = _make_client(auth_header="X-API-Key", auth_header_value="s3cr3t")
        r = c.get("/health")
        assert r.status_code == 200

    def test_both_header_and_query_auth(self) -> None:
        c = _make_client(
            auth_header="X-API-Key", auth_header_value="hdr",
            auth_query="token", auth_query_value="tok",
        )
        meta = {"image_id": 1, "task_id": 2, "label_mapping": {}, "requested_types": ["box"]}
        # both present and correct
        r = c.post(
            "/predict?token=tok",
            files={"image": ("a.png", b"\x89PNG", "image/png")},
            data={"metadata": json.dumps(meta)},
            headers={"X-API-Key": "hdr"},
        )
        assert r.status_code == 200
        # header wrong, query right → 401
        r2 = c.post(
            "/predict?token=tok",
            files={"image": ("a.png", b"", "image/png")},
            headers={"X-API-Key": "wrong"},
        )
        assert r2.status_code == 401


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_factory_equivalent(self) -> None:
        srv = create_app(_EchoPredictor(), host="127.0.0.1", port=0)
        assert isinstance(srv, InferenceServer)
        assert srv.host == "127.0.0.1"


# ---------------------------------------------------------------------------
# CLI import resolve
# ---------------------------------------------------------------------------


class TestCLIImport:
    def test_from_module_class(self) -> None:
        pred = _import_predictor("anno_sdk.handler:Predictor")
        assert isinstance(pred, Predictor)

    def test_with_trailing_call_parens(self) -> None:
        pred = _import_predictor("anno_sdk.handler:Predictor()")
        assert isinstance(pred, Predictor)

    def test_from_module_instance(self) -> None:
        pred = _import_predictor(
            "tests.test_server:_EchoPredictor()"
        )
        assert isinstance(pred, _EchoPredictor)

    def test_invalid_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected"):
            _import_predictor("no_colon_here")
