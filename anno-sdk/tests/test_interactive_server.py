"""End-to-end tests for the interactive inference server (auth + image cache)."""

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from anno_sdk import (  # noqa: E402
    Annotation,
    BoxPrompt,
    InteractiveInferenceServer,
    InteractivePredictor,
    Polygon2D,
    SessionStore,
)


class _Model(InteractivePredictor):
    """Records how often embed runs so we can prove the image is cached once."""

    def __init__(self):
        self.embed_calls = 0

    def embed_image(self, image_bytes, meta):
        self.embed_calls += 1
        return {"size": len(image_bytes)}

    def predict(self, image_state, meta):
        # Echo the cached state size + prompt count so tests can assert reuse.
        n = len(meta.prompts)
        return Annotation(label=n, geometry=Polygon2D([[0, 0], [1, 1], [2, 0]]))


def _client(**kw):
    model = _Model()
    server = InteractiveInferenceServer(
        model,
        auth_header="X-API-Key",
        auth_header_value="s3cr3t",
        public_url="https://sam.example.com/",
        **kw,
    )
    return TestClient(server._app), model


def _open_session(client, session_id=7):
    return client.post(
        "/session",
        headers={"X-API-Key": "s3cr3t"},
        json={"session_id": session_id, "image_id": 42, "requested_types": ["polygon"]},
    )


def _upload(client, session_id, token):
    return client.post(
        f"/{session_id}/infer_image",
        headers={"X-Session-Token": token},
        files={"image": ("a.png", b"bytes", "image/png")},
    )


def test_session_requires_provider_credential():
    client, _ = _client()
    bad = client.post("/session", json={"session_id": 1, "image_id": 1})
    assert bad.status_code == 401


def test_session_mints_token_and_predict_url():
    client, _ = _client()
    res = _open_session(client)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["token"]
    assert body["expires_at"]
    assert body["session_ref"] == "7"
    # public_url is normalized (trailing slash stripped) and returned as predict_url.
    assert body["predict_url"] == "https://sam.example.com"


def test_predict_requires_valid_token():
    client, _ = _client()
    _open_session(client)
    res = client.post("/7/predict", headers={"X-Session-Token": "wrong"}, json={
        "image_id": 42, "session_id": 7, "step_index": 1, "prompts": [],
    })
    assert res.status_code == 401


def test_predict_before_image_conflict():
    client, _ = _client()
    token = _open_session(client).json()["token"]
    res = client.post(
        "/7/predict",
        headers={"X-Session-Token": token},
        json={"image_id": 42, "session_id": 7, "step_index": 1, "prompts": []},
    )
    assert res.status_code == 409


def test_full_flow_image_cached_once():
    client, model = _client()
    token = _open_session(client).json()["token"]
    headers = {"X-Session-Token": token}

    # Upload the image once.
    up = client.post(
        "/7/infer_image",
        headers=headers,
        files={"image": ("a.png", b"\x89PNG-bytes", "image/png")},
        data={"metadata": json.dumps({"width": 64, "height": 64})},
    )
    assert up.status_code == 200, up.text
    assert up.json()["status"] == "cached"

    # Two prompt steps reuse the cached embedding — embed_image runs only once.
    for i in range(2):
        step = client.post(
            "/7/predict",
            headers=headers,
            json={
                "image_id": 42,
                "session_id": 7,
                "step_index": i + 1,
                "prompts": [{"type": "positive_point", "x": 1, "y": 2}],
            },
        )
        assert step.status_code == 200, step.text
        body = step.json()
        assert body["annotation"]["annotation_type"] == "polygon"
        assert body["annotation"]["label"] == 1  # one prompt

    assert model.embed_calls == 1


def test_infer_image_requires_token():
    client, _ = _client()
    _open_session(client)
    res = client.post(
        "/7/infer_image",
        files={"image": ("a.png", b"bytes", "image/png")},
    )
    assert res.status_code == 401


def test_delete_session_frees_state():
    client, _ = _client()
    token = _open_session(client).json()["token"]
    headers = {"X-Session-Token": token}
    client.post(
        "/7/infer_image",
        headers=headers,
        files={"image": ("a.png", b"bytes", "image/png")},
    )
    assert client.delete("/7", headers=headers).status_code == 200
    # After deletion the token no longer authenticates.
    res = client.post(
        "/7/predict",
        headers=headers,
        json={"image_id": 42, "session_id": 7, "step_index": 1, "prompts": []},
    )
    assert res.status_code == 401


def test_complete_requires_provider_credential():
    client, _ = _client()
    _open_session(client, session_id=7)
    # No provider credential -> 401.
    assert client.post("/session/7/complete").status_code == 401


def test_predict_receives_typed_prompts():
    captured = {}

    class _TypedModel(InteractivePredictor):
        def embed_image(self, image_bytes, meta):
            return b"state"

        def predict(self, image_state, meta):
            captured["prompts"] = meta.prompts
            return None

    server = InteractiveInferenceServer(
        _TypedModel(),
        auth_header="X-API-Key",
        auth_header_value="s3cr3t",
        public_url="https://x/",
    )
    client = TestClient(server._app)
    token = _open_session(client, session_id=7).json()["token"]
    headers = {"X-Session-Token": token}
    _upload(client, 7, token)
    res = client.post(
        "/7/predict",
        headers=headers,
        json={
            "image_id": 42,
            "session_id": 7,
            "step_index": 1,
            "prompts": [{"type": "box", "x": 10, "y": 20, "width": 30, "height": 40}],
        },
    )
    assert res.status_code == 200, res.text
    assert isinstance(captured["prompts"][0], BoxPrompt)
    assert captured["prompts"][0].width == 30
