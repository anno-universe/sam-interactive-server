"""Tests for DO serialization / deserialization."""

from __future__ import annotations

import datetime

import pytest

from anno_sdk.types import (
    Annotation,
    AnnotationBatchResult,
    AnnotationModifyResult,
    AnnotationResultItem,
    Box2D,
    Image,
    Keypoint2D,
    Polygon2D,
    PaginatedResponse,
    ProjectMeta,
    RotatedBox2D,
)

# ---------------------------------------------------------------------------
# Geometry DOs — to_dict
# ---------------------------------------------------------------------------


class TestBox2D:
    def test_to_dict_includes_rotation_zero(self) -> None:
        box = Box2D(x=10.5, y=20.1, width=100.0, height=50.0)
        assert box.to_dict() == {
            "x": 10.5,
            "y": 20.1,
            "width": 100.0,
            "height": 50.0,
            "rotation": 0.0,
        }

    def test_annotation_type_is_box(self) -> None:
        assert Box2D(0, 0, 1, 1).annotation_type == "box"


class TestRotatedBox2D:
    def test_to_dict_includes_rotation(self) -> None:
        box = RotatedBox2D(x=5.0, y=5.0, width=20.0, height=30.0, rotation=45.0)
        assert box.to_dict() == {
            "x": 5.0,
            "y": 5.0,
            "width": 20.0,
            "height": 30.0,
            "rotation": 45.0,
        }

    def test_annotation_type_is_box(self) -> None:
        assert RotatedBox2D(0, 0, 1, 1, 0).annotation_type == "box"


class TestPolygon2D:
    def test_to_dict(self) -> None:
        poly = Polygon2D(points=[[0, 0], [10, 0], [10, 10], [0, 10]])
        assert poly.to_dict() == {"points": [[0, 0], [10, 0], [10, 10], [0, 10]]}

    def test_annotation_type_is_polygon(self) -> None:
        assert Polygon2D([]).annotation_type == "polygon"


class TestPolygon2DFromBinaryMask:
    """Tests for Polygon2D.from_binary_mask()."""

    def test_simple_square(self) -> None:
        import numpy as np

        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 3:7] = 1  # 4x4 square

        poly = Polygon2D.from_binary_mask(mask)
        assert len(poly.points) >= 4
        # All points should be within the mask bounds
        for x, y in poly.points:
            assert 2 <= y < 6
            assert 3 <= x < 7

    def test_accepts_list_of_lists(self) -> None:
        mask = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        poly = Polygon2D.from_binary_mask(mask)
        assert len(poly.points) >= 3

    def test_empty_mask_raises(self) -> None:
        import numpy as np

        with pytest.raises(ValueError, match="empty"):
            Polygon2D.from_binary_mask(np.zeros((5, 5), dtype=np.uint8))

    def test_wrong_ndim_raises(self) -> None:
        import numpy as np

        with pytest.raises(ValueError, match="2-D"):
            Polygon2D.from_binary_mask(np.zeros((3, 3, 3), dtype=np.uint8))

    def test_annotation_type(self) -> None:
        import numpy as np

        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1, 1] = 1
        poly = Polygon2D.from_binary_mask(mask)
        assert poly.annotation_type == "polygon"


class TestKeypoint2D:
    def test_to_dict(self) -> None:
        kp = Keypoint2D(points=[[1.5, 2.5], [3.0, 4.0]])
        assert kp.to_dict() == {"points": [[1.5, 2.5], [3.0, 4.0]]}

    def test_annotation_type_is_keypoint(self) -> None:
        assert Keypoint2D([]).annotation_type == "keypoint"


# ---------------------------------------------------------------------------
# Annotation payload
# ---------------------------------------------------------------------------


class TestAnnotation:
    def test_to_dict_box2d(self) -> None:
        ann = Annotation(label=3, geometry=Box2D(0, 0, 100, 50))
        assert ann.to_dict() == {
            "annotation_type": "box",
            "label": 3,
            "box": {"x": 0, "y": 0, "width": 100, "height": 50, "rotation": 0.0},
        }

    def test_to_dict_rotated_box2d(self) -> None:
        ann = Annotation(label=1, geometry=RotatedBox2D(0, 0, 10, 10, 30.0))
        assert ann.to_dict() == {
            "annotation_type": "box",
            "label": 1,
            "box": {"x": 0, "y": 0, "width": 10, "height": 10, "rotation": 30.0},
        }

    def test_to_dict_polygon2d(self) -> None:
        ann = Annotation(label=None, geometry=Polygon2D([[0, 0], [1, 1]]))
        assert ann.to_dict() == {
            "annotation_type": "polygon",
            "label": None,
            "polygon": {"points": [[0, 0], [1, 1]]},
        }

    def test_to_dict_keypoint2d(self) -> None:
        ann = Annotation(label=0, geometry=Keypoint2D([[5.0, 5.0]]))
        assert ann.to_dict() == {
            "annotation_type": "keypoint",
            "label": 0,
            "keypoint": {"points": [[5.0, 5.0]]},
        }

    def test_to_dict_includes_client_ref(self) -> None:
        ann = Annotation(
            label=2, geometry=Box2D(0, 0, 10, 10), client_ref="ref-001"
        )
        d = ann.to_dict()
        assert d["client_ref"] == "ref-001"

    def test_to_dict_omits_client_ref_when_none(self) -> None:
        ann = Annotation(label=2, geometry=Box2D(0, 0, 10, 10))
        assert "client_ref" not in ann.to_dict()

    def test_from_geometry_convenience(self) -> None:
        ann = Annotation.from_geometry(Box2D(0, 0, 1, 1), label=5, client_ref="x")
        assert ann.label == 5
        assert isinstance(ann.geometry, Box2D)
        assert ann.client_ref == "x"


# ---------------------------------------------------------------------------
# Response DOs — from_dict
# ---------------------------------------------------------------------------


class TestImage:
    def test_from_dict(self) -> None:
        data = {
            "id": 1,
            "file_name": "cat.jpg",
            "width": 640,
            "height": 480,
            "file_url": "http://localhost:8000/api/infers/project/images/1/original_file",
        }
        img = Image.from_dict(data)
        assert img.id == 1
        assert img.file_name == "cat.jpg"
        assert img.width == 640
        assert img.height == 480
        assert "original_file" in img.file_url

    def test_from_dict_width_height_nullable(self) -> None:
        data = {
            "id": 2,
            "file_name": "dog.png",
            "width": None,
            "height": None,
            "file_url": "http://localhost/file",
        }
        img = Image.from_dict(data)
        assert img.width is None
        assert img.height is None


class TestProjectMeta:
    def test_from_dict(self) -> None:
        data = {
            "id": 42,
            "name": "My Dataset",
            "description": "A test dataset",
            "meta_info": {"camera": "iPhone"},
            "label_mapping": {"cat": 0, "dog": 1},
            "created_at": "2025-01-15T10:30:00Z",
            "updated_at": "2025-06-01T12:00:00+08:00",
        }
        meta = ProjectMeta.from_dict(data)
        assert meta.id == 42
        assert meta.name == "My Dataset"
        assert meta.label_mapping == {"cat": 0, "dog": 1}
        assert meta.created_at.year == 2025
        assert meta.created_at.month == 1
        assert meta.updated_at.year == 2025


class TestPaginatedResponse:
    def test_from_dict_with_factory(self) -> None:
        data = {
            "count": 2,
            "limit": 100,
            "offset": 0,
            "items": [
                {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100, "file_url": "http://x"},
                {"id": 2, "file_name": "b.jpg", "width": 200, "height": 200, "file_url": "http://y"},
            ],
        }
        page = PaginatedResponse.from_dict(data, item_factory=Image.from_dict)
        assert page.count == 2
        assert page.limit == 100
        assert len(page.items) == 2
        assert page.items[0].id == 1
        assert page.items[1].file_name == "b.jpg"


class TestAnnotationBatchResult:
    def test_from_dict_mixed_results(self) -> None:
        data = {
            "created": 2,
            "failed": 1,
            "results": [
                {"client_ref": "r1", "image_id": 1, "annotation_id": 10, "status": "created"},
                {"client_ref": "r2", "image_id": 1, "annotation_id": 11, "status": "created"},
                {
                    "client_ref": "r3",
                    "image_id": 1,
                    "annotation_id": None,
                    "status": "error",
                    "error": "Invalid geometry",
                },
            ],
        }
        result = AnnotationBatchResult.from_dict(data)
        assert result.created == 2
        assert result.failed == 1
        assert len(result.results) == 3

        ok = result.results[0]
        assert ok.is_success
        assert ok.annotation_id == 10

        err = result.results[2]
        assert not err.is_success
        assert err.status == "error"
        assert err.error == "Invalid geometry"


class TestAnnotationResultItem:
    def test_is_success_for_created(self) -> None:
        item = AnnotationResultItem(
            client_ref="x", image_id=1, annotation_id=5, status="created"
        )
        assert item.is_success

    def test_is_success_for_error(self) -> None:
        item = AnnotationResultItem(
            client_ref="x", image_id=1, annotation_id=None, status="error", error="boom"
        )
        assert not item.is_success


class TestAnnotationModifyResult:
    def test_from_dict(self) -> None:
        data = {
            "id": 99,
            "image_id": 1,
            "annotation_type": "box",
            "label": 2,
            "data": {"x": 1, "y": 2, "width": 3, "height": 4, "rotation": 0.0},
            "is_active": True,
            "created_at": "2025-06-15T08:00:00Z",
            "modified_at": "2025-06-15T08:00:00Z",
        }
        result = AnnotationModifyResult.from_dict(data)
        assert result.id == 99
        assert result.annotation_type == "box"
        assert result.label == 2
        assert result.is_active is True
        assert isinstance(result.created_at, datetime.datetime)
