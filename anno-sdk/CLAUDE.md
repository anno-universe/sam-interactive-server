# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`anno-sdk` is a Python client library for the Anno annotation platform's inference API (`anno_infers` module in `~/Code/anno`). It lets AI/ML model servers authenticate with a per-project API key, paginate/iterate over images, and submit or modify annotations.

The backend this SDK targets lives at `~/Code/anno` — a Django 6.0 + django-ninja-extra application. All relevant backend context is summarized below.

## Backend Reference (anno project at ~/Code/anno)

### API surface the SDK wraps

All endpoints are under `{base_url}/api/infers/project` and authenticate via `X-API-Key` header (Flow A — project API key, not user JWT).

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/meta` | Project metadata + `label_mapping` dict |
| `GET` | `/images?limit=&offset=&has_active_annotations=` | Paginated image list (offset/limit, max 500) |
| `GET` | `/images/{id}` | Single image detail |
| `GET` | `/images/{id}/original_file` | Image bytes (streaming or 307 redirect) |
| `POST` | `/images/{id}/annotations` | Submit batch annotations for one image |
| `PATCH` | `/images/{id}/annotations/{annotation_id}` | Modify an existing annotation (immutable pattern) |

### Pagination response shape

```python
# PaginatedResponse[ItemT]
{"count": int, "limit": int, "offset": int, "items": list[ItemT]}
```

### Image response shape

```python
{"id": int, "file_name": str, "width": int|null, "height": int|null, "file_url": str}
```

### Annotation submission (POST body)

```json
{
  "annotations": [
    {
      "annotation_type": "polygon" | "box" | "keypoint",
      "label": int | null,
      "polygon": {"points": [[x1,y1], ...]} | null,
      "box": {"x": float, "y": float, "width": float, "height": float, "rotation": float} | null,
      "keypoint": {"points": [[x1,y1], ...]} | null,
      "client_ref": str | null
    }
  ]
}
```

### Annotation batch response

```json
{
  "created": int,
  "failed": int,
  "results": [
    {"client_ref": str|null, "image_id": int, "annotation_id": int|null, "status": "created"|"error", "error": str|null}
  ]
}
```

### Annotation modify (PATCH) — immutable pattern

PATCH body is the same shape as a single submission item (minus `client_ref`). The backend creates a **new** `Annotation2D` row, deactivates the old one (`is_active=False`), and records an `Operation(action="modify")`.

### Annotation types (backend models, `~/Code/anno/anno_images/models.py`)

- **`Annotation2D`** — header table: `id`, `image_id`, `project_id`, `annotation_type` (`"polygon"`|`"box"`|`"keypoint"`), `label` (int|null), `is_active`
- **`Polygon2D`** — 1:1 with Annotation2D, field `points` (JSON `[[x,y], ...]`)
- **`Box2D`** — 1:1 with Annotation2D, fields `x`, `y`, `width`, `height`, `rotation` (degrees, default 0.0). Rotation makes it functionally a rotated box.
- **`Keypoint2D`** — 1:1 with Annotation2D, field `points` (JSON `[[x,y], ...]`)

### Immutable annotation pattern

All annotation mutations create new rows; existing rows are never updated in place.
- **Create:** new `Annotation2D` + subtype row + `Operation(action="add")`
- **Modify:** new `Annotation2D` + subtype row; old gets `is_active=False`; `Operation(action="modify")` links old→new
- **Delete:** set `is_active=False`; `Operation(action="delete")`

## SDK Design

### Package layout (planned)

```
anno_sdk/
    __init__.py          # exports Client, DO classes
    client.py            # Client class — main entry point
    types.py             # DO (data object) classes: Box2D, RotatedBox2D, Mask2D, Keypoint2D
    pagination.py        # PaginatedResponse, paginate_images, iter_images helpers
    exceptions.py        # SDK-specific exceptions
```

### Client

```python
class Client:
    def __init__(self, base_url: str, api_key: str): ...
    def get_meta(self) -> ProjectMeta: ...
    def paginate_images(self, *, limit=100, offset=0, has_active_annotations=None) -> PaginatedResponse[Image]: ...
    def iter_images(self, *, limit=100, has_active_annotations=None) -> Iterator[Image]: ...
    def get_image(self, image_id: int) -> Image: ...
    def get_image_file(self, image_id: int) -> bytes: ...
    def upload_annotations(self, image_id: int, annotations: list[AnnotationDO]) -> AnnotationBatchResult: ...
    def modify_annotation(self, image_id: int, annotation_id: int, annotation: AnnotationDO) -> AnnotationModifyResult: ...
```

- `base_url` is the root of the anno server (e.g. `http://localhost:8000`).
- `api_key` is the plaintext project API key (`ak_xxxxxxxx.yyyyy...`).
- Client sends `X-API-Key` header on every request.
- Use `httpx` or `requests` for HTTP (prefer `httpx` — modern, async-capable, good typing).

### DO (Data Object) classes

Mapped to backend annotation geometry types. The `Box2D` backend model has a `rotation` field, so both axis-aligned and rotated boxes share one wire format. The SDK should expose two ergonomic classes that both serialize to the backend's `box` shape:

```python
@dataclass
class Box2D:
    """Axis-aligned bounding box."""
    x: float
    y: float
    width: float
    height: float
    def to_dict(self) -> dict: ...  # {"x": ..., "y": ..., "width": ..., "height": ..., "rotation": 0.0}

@dataclass
class RotatedBox2D:
    """Rotated bounding box."""
    x: float
    y: float
    width: float
    height: float
    rotation: float  # degrees clockwise
    def to_dict(self) -> dict: ...  # {"x": ..., "y": ..., "width": ..., "height": ..., "rotation": ...}

@dataclass
class Mask2D:
    """Polygon mask."""
    points: list[list[float]]  # [[x1,y1], [x2,y2], ...]
    def to_dict(self) -> dict: ...  # {"points": [[x1,y1], ...]}

@dataclass
class Keypoint2D:
    """Keypoint set."""
    points: list[list[float]]  # [[x1,y1], [x2,y2], ...]
    def to_dict(self) -> dict: ...  # {"points": [[x1,y1], ...]}
```

A union type or base class for annotation payloads:

```python
AnnotationDO = Box2D | RotatedBox2D | Mask2D | Keypoint2D
```

When uploading, each annotation also carries `label: int | None` and an `annotation_type` string. The `upload_annotations` method should accept a list of objects that combine a geometry DO with a label, e.g.:

```python
@dataclass
class Annotation:
    label: int | None
    geometry: AnnotationDO
    client_ref: str | None = None
```

The client serializes this to the backend's wire format (`annotation_type`, `label`, and exactly one of `polygon`/`box`/`keypoint`).

### `iter_images` vs `paginate_images`

- `paginate_images` — low-level: call with explicit limit/offset, returns a `PaginatedResponse[Image]` with `count`, `items`, etc.
- `iter_images` — high-level: wraps `paginate_images` in a generator, auto-advancing offset until exhausted. Accepts `limit` for page size (how many per HTTP request), not total.

### Error handling

- HTTP errors (4xx, 5xx) → raise an SDK-specific exception (e.g. `AnnoAPIError` with status code and body).
- Network errors → raise a connection-level exception.
- Per-item errors in batch annotation uploads are NOT raised — they're reported in `AnnotationBatchResult.results` with `status="error"`.

### Target Python version

Match or slightly trail the backend: Python >=3.12 (backend is 3.14, but SDK should be broadly usable). Confirm with user.

## Development Commands (proposed)

Once the Python package is bootstrapped:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test
pytest tests/test_client.py::test_paginate_images

# Lint
ruff check .

# Format
ruff format .

# Type check
mypy anno_sdk/
```

The build system should be `pyproject.toml` with `hatchling` or `setuptools`. Dependencies: `httpx`, `python-dateutil` (for timestamp parsing). Dev dependencies: `pytest`, `ruff`, `mypy`, `responses` or `pytest-httpx` for HTTP mocking.
