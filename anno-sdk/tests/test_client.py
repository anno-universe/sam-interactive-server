"""Tests for the Client class with HTTP mocking."""

from __future__ import annotations

import re

import httpx
import pytest
from pytest_httpx import HTTPXMock

from anno_sdk import (
    AnnoAPIError,
    AnnoConnectionError,
    Annotation,
    Box2D,
    Client,
    Image,
    Keypoint2D,
    Polygon2D,
    PaginatedResponse,
    ProjectMeta,
    RotatedBox2D,
)

BASE_URL = "http://anno.example.com"
API_KEY = "ak_deadbeef.secretsecretsecretsecretsecret"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> Client:
    return Client(base_url=BASE_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
# GET /meta
# ---------------------------------------------------------------------------


META_RESPONSE = {
    "id": 1,
    "name": "Test Project",
    "description": "A test",
    "meta_info": {},
    "label_mapping": {"cat": 0, "dog": 1},
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-06-01T12:00:00Z",
}


def test_get_meta(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/meta",
        json=META_RESPONSE,
    )
    meta = client.get_meta()
    assert isinstance(meta, ProjectMeta)
    assert meta.name == "Test Project"
    assert meta.label_mapping == {"cat": 0, "dog": 1}


def test_get_meta_unauthorized(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/meta",
        status_code=401,
        text="Unauthorized",
    )
    with pytest.raises(AnnoAPIError) as exc:
        client.get_meta()
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# GET /images
# ---------------------------------------------------------------------------


_F1 = f"{BASE_URL}/api/infers/project/images/1/original_file"
_F2 = f"{BASE_URL}/api/infers/project/images/2/original_file"
IMAGE_ITEMS = [
    {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480, "file_url": _F1},
    {"id": 2, "file_name": "b.jpg", "width": 800, "height": 600, "file_url": _F2},
]


def test_paginate_images(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(BASE_URL)}/api/infers/project/images(\?.*)?$"),
        json={"count": 2, "limit": 100, "offset": 0, "items": IMAGE_ITEMS},
    )
    page = client.paginate_images()
    assert isinstance(page, PaginatedResponse)
    assert page.count == 2
    assert len(page.items) == 2
    assert page.items[0].id == 1
    assert isinstance(page.items[0], Image)


def test_paginate_images_with_filter(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(BASE_URL)}/api/infers/project/images(\?.*)?$"),
        json={"count": 0, "limit": 50, "offset": 10, "items": []},
    )
    page = client.paginate_images(limit=50, offset=10, has_active_annotations=True)
    assert page.count == 0
    # Verify query params were sent
    req = httpx_mock.get_request()
    assert req is not None
    assert "has_active_annotations=true" in str(req.url)


def test_paginate_images_exclude_annotated(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(BASE_URL)}/api/infers/project/images(\?.*)?$"),
        json={"count": 0, "limit": 100, "offset": 0, "items": []},
    )
    client.paginate_images(has_active_annotations=False)
    req = httpx_mock.get_request()
    assert "has_active_annotations=false" in str(req.url)


# ---------------------------------------------------------------------------
# iter_images
# ---------------------------------------------------------------------------


_IMAGES_URL = re.compile(rf"^{re.escape(BASE_URL)}/api/infers/project/images(\?.*)?$")


def test_iter_images_single_page(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=_IMAGES_URL,
        json={"count": 2, "limit": 100, "offset": 0, "items": IMAGE_ITEMS},
    )
    images = list(client.iter_images())
    assert len(images) == 2
    assert images[0].file_name == "a.jpg"
    assert images[1].file_name == "b.jpg"


def test_iter_images_multi_page(client: Client, httpx_mock: HTTPXMock) -> None:
    # Page 1
    httpx_mock.add_response(
        url=_IMAGES_URL,
        json={"count": 3, "limit": 2, "offset": 0, "items": IMAGE_ITEMS},
    )
    # Page 2
    httpx_mock.add_response(
        url=_IMAGES_URL,
        json={
            "count": 3,
            "limit": 2,
            "offset": 2,
            "items": [
                {
                    "id": 3,
                    "file_name": "c.jpg",
                    "width": 100,
                    "height": 100,
                    "file_url": f"{BASE_URL}/api/infers/project/images/3/original_file",
                },
            ],
        },
    )
    images = list(client.iter_images(limit=2))
    assert len(images) == 3
    assert [img.file_name for img in images] == ["a.jpg", "b.jpg", "c.jpg"]


def test_iter_images_empty(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=_IMAGES_URL,
        json={"count": 0, "limit": 100, "offset": 0, "items": []},
    )
    assert list(client.iter_images()) == []


# ---------------------------------------------------------------------------
# GET /images/{id}
# ---------------------------------------------------------------------------


def test_get_image(client: Client, httpx_mock: HTTPXMock) -> None:
    _fu = f"{BASE_URL}/api/infers/project/images/42/original_file"
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/images/42",
        json={
            "id": 42,
            "file_name": "cat.png",
            "width": 300,
            "height": 200,
            "file_url": _fu,
        },
    )
    img = client.get_image(42)
    assert img.id == 42
    assert img.file_name == "cat.png"


def test_get_image_not_found(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/api/infers/project/images/999", status_code=404)
    with pytest.raises(AnnoAPIError) as exc:
        client.get_image(999)
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /images/{id}/original_file
# ---------------------------------------------------------------------------


def test_get_image_file(client: Client, httpx_mock: HTTPXMock) -> None:
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/images/1/original_file",
        content=png_bytes,
        headers={"content-type": "image/png"},
    )
    data = client.get_image_file(1)
    assert data == png_bytes


def test_get_image_file_error(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/images/1/original_file",
        status_code=403,
    )
    with pytest.raises(AnnoAPIError):
        client.get_image_file(1)


# ---------------------------------------------------------------------------
# POST /images/{id}/annotations
# ---------------------------------------------------------------------------


BATCH_RESPONSE = {
    "created": 2,
    "failed": 1,
    "results": [
        {"client_ref": "r1", "image_id": 1, "annotation_id": 100, "status": "created"},
        {"client_ref": "r2", "image_id": 1, "annotation_id": 101, "status": "created"},
        {
            "client_ref": "r3",
            "image_id": 1,
            "annotation_id": None,
            "status": "error",
            "error": "bad geometry",
        },
    ],
}


def test_upload_annotations(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/images/1/annotations",
        json=BATCH_RESPONSE,
    )
    annotations = [
        Annotation.from_geometry(Box2D(0, 0, 10, 10), label=0, client_ref="r1"),
        Annotation.from_geometry(Box2D(10, 0, 20, 10), label=1, client_ref="r2"),
        Annotation.from_geometry(Polygon2D([[0, 0]]), label=None, client_ref="r3"),
    ]
    result = client.upload_annotations(image_id=1, annotations=annotations)
    assert result.created == 2
    assert result.failed == 1
    assert len(result.results) == 3
    assert result.results[0].is_success
    assert not result.results[2].is_success

    # Verify the request body was serialized correctly
    body = httpx_mock.get_request().read().decode()
    assert '"annotation_type"' in body
    assert '"box"' in body
    assert '"polygon"' in body


def test_upload_annotations_all_geometry_types(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/images/1/annotations",
        json={"created": 3, "failed": 0, "results": []},
    )
    annotations = [
        Annotation.from_geometry(Box2D(0, 0, 1, 1), label=0),
        Annotation.from_geometry(RotatedBox2D(0, 0, 1, 1, 30), label=1),
        Annotation.from_geometry(Polygon2D([[0, 0]]), label=2),
        Annotation.from_geometry(Keypoint2D([[1, 1]]), label=3),
    ]
    result = client.upload_annotations(image_id=1, annotations=annotations)
    assert result.created == 3

    # Verify body contains all annotation types
    body = httpx_mock.get_request().read().decode()
    assert '"box"' in body
    assert '"polygon"' in body
    assert '"keypoint"' in body


# ---------------------------------------------------------------------------
# PATCH /images/{id}/annotations/{aid}
# ---------------------------------------------------------------------------


MODIFY_RESPONSE = {
    "id": 200,
    "image_id": 1,
    "annotation_type": "box",
    "label": 1,
    "data": {"x": 5, "y": 5, "width": 50, "height": 50, "rotation": 0.0},
    "is_active": True,
    "created_at": "2025-06-15T08:00:00Z",
    "modified_at": "2025-06-15T08:01:00Z",
}


def test_modify_annotation(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/images/1/annotations/100",
        json=MODIFY_RESPONSE,
    )
    ann = Annotation.from_geometry(Box2D(5, 5, 50, 50), label=1)
    result = client.modify_annotation(image_id=1, annotation_id=100, annotation=ann)
    assert result.id == 200
    assert result.annotation_type == "box"
    assert result.is_active is True

    # Verify the body
    req = httpx_mock.get_request()
    assert req.method == "PATCH"


def test_modify_annotation_not_found(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/images/1/annotations/999",
        status_code=404,
    )
    ann = Annotation.from_geometry(Box2D(0, 0, 1, 1), label=0)
    with pytest.raises(AnnoAPIError) as exc:
        client.modify_annotation(image_id=1, annotation_id=999, annotation=ann)
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_network_error_triggers_connection_error(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
    with pytest.raises(AnnoConnectionError):
        client.get_meta()


def test_timeout_triggers_connection_error(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("timeout"))
    with pytest.raises(AnnoConnectionError):
        client.get_meta()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager() -> None:
    c = Client(base_url=BASE_URL, api_key=API_KEY)
    assert not c._http.is_closed
    with c:
        pass
    assert c._http.is_closed


def test_explicit_close(client: Client) -> None:
    assert not client._http.is_closed
    client.close()
    assert client._http.is_closed


# ---------------------------------------------------------------------------
# Header / auth
# ---------------------------------------------------------------------------


def test_api_key_sent_in_header(client: Client, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/api/infers/project/meta",
        json=META_RESPONSE,
    )
    client.get_meta()
    req = httpx_mock.get_request()
    assert req is not None
    assert req.headers["X-API-Key"] == API_KEY
