"""Data-object classes for the Anno inference SDK.

Geometry DOs
------------
Each geometry class knows its own ``annotation_type`` string and can
serialize itself to the backend wire format via :meth:`to_dict`.

* :class:`Polygon2D` — polygon defined by ``[[x, y], ...]`` points, backend type ``"polygon"``
* :class:`Box2D` — axis-aligned bounding box, backend type ``"box"``
* :class:`RotatedBox2D` — rotated bounding box, backend type ``"box"``
* :class:`Keypoint2D` — keypoint set, backend type ``"keypoint"``

Payload wrapper
---------------
:class:`Annotation` bundles a geometry DO with a class label and an
optional client-side reference.  Its :meth:`Annotation.to_dict` produces
the exact payload the backend expects for a single annotation item.

Response / result DOs
---------------------
:class:`Image`, :class:`ProjectMeta`, :class:`PaginatedResponse`,
:class:`AnnotationBatchResult`, :class:`AnnotationResultItem`,
:class:`AnnotationModifyResult`.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Geometry data objects
# ---------------------------------------------------------------------------


@dataclass
class Box2D:
    """Axis-aligned bounding box.

    Serializes to the backend ``"box"`` geometry with ``rotation=0.0``.
    """

    x: float
    y: float
    width: float
    height: float

    @property
    def annotation_type(self) -> str:
        return "box"

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "rotation": 0.0,
        }


@dataclass
class RotatedBox2D:
    """Rotated bounding box.

    Serializes to the backend ``"box"`` geometry including the rotation
    angle in degrees clockwise.
    """

    x: float
    y: float
    width: float
    height: float
    rotation: float

    @property
    def annotation_type(self) -> str:
        return "box"

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
        }


@dataclass
class Polygon2D:
    """Polygon defined by a closed list of ``[x, y]`` points."""

    points: list[list[float]]

    @property
    def annotation_type(self) -> str:
        return "polygon"

    def to_dict(self) -> dict:
        return {"points": self.points}

    @classmethod
    def from_binary_mask(
        cls,
        mask,
        *,
        simplification_epsilon: float | None = None,
    ) -> Polygon2D:
        """Create a ``Polygon2D`` from a 2-D binary segmentation mask.

        Extracts the largest external contour from the mask and returns it as
        polygon points.  The mask is expected as a 2-D array where non-zero
        values indicate the region of interest.

        Parameters
        ----------
        mask:
            A 2-D array-like (e.g. ``numpy.ndarray``, ``list[list[int]]``).
        simplification_epsilon:
            If provided, simplify the contour with the Douglas-Peucker
            algorithm (requires ``opencv-python`` / ``cv2``).  Larger values
            produce fewer vertices.

        Returns
        -------
        Polygon2D
            Polygon whose ``points`` are the contour vertices as
            ``[[x, y], ...]`` in pixel coordinates.

        Raises
        ------
        ImportError
            If ``numpy`` is not installed.
        ValueError
            If the mask is empty (all zeros) or has no contour.
        """
        import numpy as np

        mask_arr = np.asarray(mask, dtype=np.uint8)
        if mask_arr.ndim != 2:
            raise ValueError(
                f"Expected a 2-D mask, got {mask_arr.ndim}-D array with shape {mask_arr.shape}"
            )
        if not mask_arr.any():
            raise ValueError("Mask is empty (all zeros) — no contour to extract.")

        contour = _extract_largest_contour(mask_arr)

        if contour is None or len(contour) < 3:
            raise ValueError("Failed to extract a valid contour from the mask.")

        # contour shape is (N, 1, 2) from cv2 or (N, 2) from the pure-Python
        # fallback — normalise to (N, 2).
        contour = np.squeeze(contour, axis=1) if contour.ndim == 3 else contour

        if simplification_epsilon is not None:
            contour = _simplify_contour(contour, simplification_epsilon)

        points: list[list[float]] = contour.astype(float).tolist()
        return cls(points=points)


@dataclass
class Keypoint2D:
    """Keypoint set defined by a list of ``[x, y]`` points."""

    points: list[list[float]]

    @property
    def annotation_type(self) -> str:
        return "keypoint"

    def to_dict(self) -> dict:
        return {"points": self.points}


# ---------------------------------------------------------------------------
# Union type for annotation geometry
# ---------------------------------------------------------------------------

GeometryDO = Box2D | RotatedBox2D | Polygon2D | Keypoint2D

# ---------------------------------------------------------------------------
# Annotation payload
# ---------------------------------------------------------------------------


@dataclass
class Annotation:
    """A single annotation to be uploaded or used as a modify payload.

    Bundles a geometry DO (which determines ``annotation_type``) with an
    optional integer class label and an optional client-side reference.
    """

    label: int | None
    geometry: GeometryDO
    client_ref: str | None = None

    def to_dict(self) -> dict:
        """Serialize to the backend wire format for a single annotation item."""
        payload: dict = {
            "annotation_type": self.geometry.annotation_type,
            "label": self.label,
        }
        payload[self.geometry.annotation_type] = self.geometry.to_dict()
        if self.client_ref is not None:
            payload["client_ref"] = self.client_ref
        return payload

    @classmethod
    def from_geometry(
        cls,
        geometry: GeometryDO,
        label: int | None = None,
        client_ref: str | None = None,
    ) -> Annotation:
        """Convenience constructor: ``Annotation.from_geometry(Box2D(0,0,10,10), label=1)``."""
        return cls(label=label, geometry=geometry, client_ref=client_ref)

    @classmethod
    def from_dict(cls, data: dict) -> Annotation:
        """Inverse of :meth:`to_dict`.

        Reconstructs an :class:`Annotation` (and its geometry DO) from the
        backend wire format. A ``"box"`` payload becomes a :class:`Box2D` when
        its rotation is zero/absent, otherwise a :class:`RotatedBox2D`.
        """
        return cls(
            label=data.get("label"),
            geometry=_geometry_from_dict(data["annotation_type"], data[data["annotation_type"]]),
            client_ref=data.get("client_ref"),
        )


def _geometry_from_dict(annotation_type: str, data: dict) -> GeometryDO:
    """Build a geometry DO from a wire-format geometry payload."""
    if annotation_type == "polygon":
        return Polygon2D(points=data["points"])
    if annotation_type == "keypoint":
        return Keypoint2D(points=data["points"])
    if annotation_type == "box":
        rotation = data.get("rotation") or 0.0
        if rotation:
            return RotatedBox2D(
                x=data["x"],
                y=data["y"],
                width=data["width"],
                height=data["height"],
                rotation=rotation,
            )
        return Box2D(x=data["x"], y=data["y"], width=data["width"], height=data["height"])
    raise ValueError(f"Unknown annotation_type: {annotation_type!r}")


# Module-level alias kept for callers that prefer a free function over the
# classmethod (e.g. the server parsing inference responses).
_annotation_from_dict = Annotation.from_dict


# ---------------------------------------------------------------------------
# Response / result data objects
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass
class Image:
    """An image in a project, as returned by the inference API."""

    id: int
    file_name: str
    width: int | None
    height: int | None
    file_url: str

    @classmethod
    def from_dict(cls, data: dict) -> Image:
        return cls(
            id=data["id"],
            file_name=data["file_name"],
            width=data.get("width"),
            height=data.get("height"),
            file_url=data["file_url"],
        )


@dataclass
class ProjectMeta:
    """Project metadata, including the label mapping."""

    id: int
    name: str
    description: str
    meta_info: dict
    label_mapping: dict
    created_at: datetime.datetime
    updated_at: datetime.datetime

    @classmethod
    def from_dict(cls, data: dict) -> ProjectMeta:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            meta_info=data["meta_info"],
            label_mapping=data["label_mapping"],
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
        )


@dataclass
class PaginatedResponse(Generic[T]):
    """Generic offset/limit paginated response."""

    count: int
    limit: int
    offset: int
    items: list[T]

    @classmethod
    def from_dict(
        cls,
        data: dict,
        item_factory: callable = lambda d: d,
    ) -> PaginatedResponse[T]:
        return cls(
            count=data["count"],
            limit=data["limit"],
            offset=data["offset"],
            items=[item_factory(item) for item in data["items"]],
        )


@dataclass
class AnnotationResultItem:
    """Per-annotation result within a batch submission."""

    client_ref: str | None
    image_id: int
    annotation_id: int | None
    status: str  # "created" | "error"
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> AnnotationResultItem:
        return cls(
            client_ref=data.get("client_ref"),
            image_id=data["image_id"],
            annotation_id=data.get("annotation_id"),
            status=data["status"],
            error=data.get("error"),
        )

    @property
    def is_success(self) -> bool:
        return self.status == "created"


@dataclass
class AnnotationBatchResult:
    """Result of a batch annotation upload."""

    created: int
    failed: int
    results: list[AnnotationResultItem]

    @classmethod
    def from_dict(cls, data: dict) -> AnnotationBatchResult:
        return cls(
            created=data["created"],
            failed=data["failed"],
            results=[AnnotationResultItem.from_dict(r) for r in data["results"]],
        )


@dataclass
class AnnotationModifyResult:
    """Result of modifying an existing annotation."""

    id: int
    image_id: int
    annotation_type: str
    label: int | None
    data: dict
    is_active: bool
    created_at: datetime.datetime
    modified_at: datetime.datetime

    @classmethod
    def from_dict(cls, data: dict) -> AnnotationModifyResult:
        return cls(
            id=data["id"],
            image_id=data["image_id"],
            annotation_type=data["annotation_type"],
            label=data.get("label"),
            data=data["data"],
            is_active=data["is_active"],
            created_at=_parse_datetime(data["created_at"]),
            modified_at=_parse_datetime(data["modified_at"]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_datetime(value: str) -> datetime.datetime:
    """Parse an ISO-8601 datetime string, handling both ``Z`` and offset suffixes."""
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Contour helpers (used by Polygon2D.from_binary_mask)
# ---------------------------------------------------------------------------


def _extract_largest_contour(mask: "numpy.ndarray") -> "numpy.ndarray | None":
    """Return the largest external contour from a binary mask.

    Tries ``cv2.findContours`` first; falls back to a pure-Python
    border-following algorithm when ``cv2`` is not available.
    """
    try:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)
    except ImportError:
        pass

    # Pure-Python fallback — simple border following.
    return _trace_border_python(mask)


def _trace_border_python(mask: "numpy.ndarray") -> "numpy.ndarray | None":
    """Trace the external border of the first non-zero region using Moore-Neighbor tracing."""
    import numpy as np

    rows, cols = mask.shape

    # Find the first non-zero pixel (topmost, then leftmost).
    nonzero = np.argwhere(mask)
    if len(nonzero) == 0:
        return None

    start_r, start_c = nonzero[0]  # (row, col)

    # Moore neighborhood: 8-connected, clockwise starting from "up".
    # Directions:  0=up, 1=up-right, 2=right, 3=down-right,
    #               4=down, 5=down-left, 6=left, 7=up-left
    dr = [-1, -1, 0, 1, 1, 1, 0, -1]
    dc = [0, 1, 1, 1, 0, -1, -1, -1]

    boundary: list[tuple[int, int]] = [(start_c, start_r)]  # (x, y)

    # Start searching from direction 0 (up); backtrack direction is opposite.
    curr_r, curr_c = start_r, start_c
    search_dir = 0  # begin searching clockwise from "up"

    while True:
        found = False
        for i in range(8):
            d = (search_dir + i) % 8
            nr = curr_r + dr[d]
            nc = curr_c + dc[d]
            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc]:
                boundary.append((nc, nr))  # (x, y)
                curr_r, curr_c = nr, nc
                # Next search starts from the direction opposite to where we came from,
                # rotated one step counter-clockwise.
                search_dir = (d + 4 + 1) % 8  # opposite + one step
                found = True
                break

        if not found:
            # Isolated single-pixel blob — return the pixel's bounding box.
            nc, nr = start_c, start_r
            boundary = [
                (nc, nr),
                (nc + 1, nr),
                (nc + 1, nr + 1),
                (nc, nr + 1),
            ]
            break

        if (curr_r, curr_c) == (start_r, start_c):
            break

        # Safety limit to avoid infinite loops on pathological input.
        if len(boundary) > rows * cols:
            break

    # Return in cv2-compatible shape (N, 1, 2).
    return np.array(boundary, dtype=np.int32).reshape(-1, 1, 2)


def _simplify_contour(
    contour: "numpy.ndarray", epsilon: float
) -> "numpy.ndarray":
    """Simplify a contour using Douglas-Peucker (requires ``cv2``)."""
    try:
        import cv2
        import numpy as np

        return cv2.approxPolyDP(contour.astype(np.float32), epsilon, closed=True)
    except ImportError:
        raise ImportError(
            "simplification_epsilon requires opencv-python (cv2). "
            "Install it with: pip install opencv-python"
        ) from None
