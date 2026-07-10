"""Tests for the server-driven inference contract (inference.py, handler.py)."""

from __future__ import annotations

from anno_sdk import (
    Annotation,
    Box2D,
    InferenceRequestMeta,
    InferenceResponse,
    Keypoint2D,
    Polygon2D,
    Predictor,
    RotatedBox2D,
    serve_predict,
)
from anno_sdk.types import _annotation_from_dict

# ---------------------------------------------------------------------------
# Annotation.from_dict — inverse of to_dict
# ---------------------------------------------------------------------------


class TestAnnotationFromDict:
    def test_box_roundtrip_zero_rotation_is_box2d(self) -> None:
        ann = Annotation(label=3, geometry=Box2D(0, 0, 100, 50))
        restored = Annotation.from_dict(ann.to_dict())
        assert isinstance(restored.geometry, Box2D)
        assert restored.label == 3
        assert restored.to_dict() == ann.to_dict()

    def test_box_with_rotation_becomes_rotated_box(self) -> None:
        ann = Annotation(label=1, geometry=RotatedBox2D(0, 0, 10, 10, rotation=30.0))
        restored = Annotation.from_dict(ann.to_dict())
        assert isinstance(restored.geometry, RotatedBox2D)
        assert restored.geometry.rotation == 30.0

    def test_polygon_roundtrip(self) -> None:
        ann = Annotation(label=None, geometry=Polygon2D([[0, 0], [1, 1], [2, 0]]))
        restored = Annotation.from_dict(ann.to_dict())
        assert isinstance(restored.geometry, Polygon2D)
        assert restored.geometry.points == [[0, 0], [1, 1], [2, 0]]
        assert restored.label is None

    def test_keypoint_roundtrip(self) -> None:
        ann = Annotation(label=5, geometry=Keypoint2D([[1, 2], [3, 4]]))
        restored = Annotation.from_dict(ann.to_dict())
        assert isinstance(restored.geometry, Keypoint2D)
        assert restored.geometry.points == [[1, 2], [3, 4]]

    def test_client_ref_preserved(self) -> None:
        ann = Annotation(label=1, geometry=Box2D(0, 0, 1, 1), client_ref="abc")
        restored = Annotation.from_dict(ann.to_dict())
        assert restored.client_ref == "abc"

    def test_module_alias_matches_classmethod(self) -> None:
        d = Annotation(label=1, geometry=Box2D(0, 0, 1, 1)).to_dict()
        assert _annotation_from_dict(d).to_dict() == Annotation.from_dict(d).to_dict()


# ---------------------------------------------------------------------------
# InferenceRequestMeta
# ---------------------------------------------------------------------------


class TestInferenceRequestMeta:
    def test_roundtrip(self) -> None:
        meta = InferenceRequestMeta(
            image_id=42,
            task_id=7,
            label_mapping={"cat": 0, "dog": 1},
            requested_types=["box", "polygon"],
            width=1920,
            height=1080,
            client_ref="ref-1",
        )
        restored = InferenceRequestMeta.from_dict(meta.to_dict())
        assert restored == meta

    def test_from_dict_tolerates_missing_optionals(self) -> None:
        meta = InferenceRequestMeta.from_dict({"image_id": 1, "task_id": 2})
        assert meta.label_mapping == {}
        assert meta.requested_types == []
        assert meta.width is None and meta.client_ref is None


# ---------------------------------------------------------------------------
# InferenceResponse
# ---------------------------------------------------------------------------


class TestInferenceResponse:
    def test_roundtrip_all_geometries(self) -> None:
        resp = InferenceResponse(
            annotations=[
                Annotation(label=1, geometry=Box2D(0, 0, 10, 10)),
                Annotation(label=2, geometry=RotatedBox2D(0, 0, 10, 10, rotation=45.0)),
                Annotation(label=None, geometry=Polygon2D([[0, 0], [1, 1]])),
                Annotation(label=3, geometry=Keypoint2D([[5, 5]])),
            ],
            model_version="v1.2",
        )
        restored = InferenceResponse.from_dict(resp.to_dict())
        assert restored.model_version == "v1.2"
        assert [a.to_dict() for a in restored.annotations] == [
            a.to_dict() for a in resp.annotations
        ]

    def test_from_dict_empty(self) -> None:
        assert InferenceResponse.from_dict({}).annotations == []


# ---------------------------------------------------------------------------
# serve_predict
# ---------------------------------------------------------------------------


class TestServePredict:
    def test_end_to_end(self) -> None:
        captured: dict = {}

        def predict(image_bytes: bytes, meta: InferenceRequestMeta) -> list[Annotation]:
            captured["bytes"] = image_bytes
            captured["meta"] = meta
            return [Annotation.from_geometry(Box2D(0, 0, 10, 10), label=meta.requested_types and 1)]

        metadata = InferenceRequestMeta(
            image_id=1, task_id=2, label_mapping={}, requested_types=["box"]
        ).to_dict()

        body = serve_predict(b"\x89PNG-bytes", metadata, predict)

        assert captured["bytes"] == b"\x89PNG-bytes"
        assert isinstance(captured["meta"], InferenceRequestMeta)
        assert body["annotations"][0]["annotation_type"] == "box"
        # The body must be re-parseable by the server.
        assert InferenceResponse.from_dict(body).annotations[0].label == 1


# ---------------------------------------------------------------------------
# Predictor (class-based)
# ---------------------------------------------------------------------------


class TestPredictor:
    def test_subclass_predict_end_to_end(self) -> None:
        captured: dict = {}

        class MyModel(Predictor):
            def predict(self, image_bytes, meta):
                captured["bytes"] = image_bytes
                captured["meta"] = meta
                return [Annotation.from_geometry(Box2D(0, 0, 10, 10), label=1)]

        metadata = InferenceRequestMeta(
            image_id=1, task_id=2, label_mapping={}, requested_types=["box"]
        ).to_dict()

        body = MyModel().serve(b"\x89PNG-bytes", metadata)

        assert captured["bytes"] == b"\x89PNG-bytes"
        assert isinstance(captured["meta"], InferenceRequestMeta)
        assert body["annotations"][0]["annotation_type"] == "box"
        assert InferenceResponse.from_dict(body).annotations[0].label == 1

    def test_base_predict_not_implemented(self) -> None:
        metadata = InferenceRequestMeta(
            image_id=1, task_id=2, label_mapping={}, requested_types=[]
        ).to_dict()
        try:
            Predictor().serve(b"x", metadata)
        except NotImplementedError:
            pass
        else:  # pragma: no cover - failure path
            raise AssertionError("base Predictor.predict should raise NotImplementedError")

    def test_setup_runs_exactly_once(self) -> None:
        class Counting(Predictor):
            def __init__(self) -> None:
                self.setup_calls = 0

            def setup(self) -> None:
                self.setup_calls += 1

            def predict(self, image_bytes, meta):
                return []

        metadata = InferenceRequestMeta(
            image_id=1, task_id=2, label_mapping={}, requested_types=[]
        ).to_dict()
        model = Counting()
        model.serve(b"a", metadata)
        model.serve(b"b", metadata)
        model.serve(b"c", metadata)
        assert model.setup_calls == 1

    def test_setup_runs_without_super_init(self) -> None:
        # Subclass overrides __init__ but does not call super().__init__().
        class NoSuper(Predictor):
            def __init__(self) -> None:
                self.loaded = False

            def setup(self) -> None:
                self.loaded = True

            def predict(self, image_bytes, meta):
                return []

        metadata = InferenceRequestMeta(
            image_id=1, task_id=2, label_mapping={}, requested_types=[]
        ).to_dict()
        model = NoSuper()
        model.serve(b"a", metadata)
        assert model.loaded is True

    def test_pre_and_post_process_hooks_flow_through(self) -> None:
        order: list[str] = []

        class Hooked(Predictor):
            def preprocess(self, image_bytes, meta):
                order.append("pre")
                return image_bytes.decode()  # bytes -> str inputs

            def predict(self, inputs, meta):
                order.append("predict")
                assert inputs == "img"  # received preprocess output, not raw bytes
                return [Annotation(label=1, geometry=Box2D(0, 0, 1, 1))]

            def postprocess(self, annotations, meta):
                order.append("post")
                # append a second annotation to prove the output is used
                return [*annotations, Annotation(label=2, geometry=Box2D(0, 0, 2, 2))]

        metadata = InferenceRequestMeta(
            image_id=1, task_id=2, label_mapping={}, requested_types=[]
        ).to_dict()
        body = Hooked().serve(b"img", metadata)

        assert order == ["pre", "predict", "post"]
        labels = [a.label for a in InferenceResponse.from_dict(body).annotations]
        assert labels == [1, 2]
