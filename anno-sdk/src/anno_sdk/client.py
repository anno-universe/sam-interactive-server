"""Anno inference API client."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from .exceptions import AnnoAPIError, AnnoConnectionError
from .types import (
    Annotation,
    AnnotationBatchResult,
    AnnotationModifyResult,
    Image,
    PaginatedResponse,
    ProjectMeta,
)

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Client:
    """HTTP client for the Anno project inference API.

    Authenticates with a per-project API key via the ``X-API-Key`` header.
    All methods return deserialized data-objects, not raw dicts.

    Supports use as a context manager::

        with Client(base_url="http://localhost:8000", api_key="ak_...") as client:
            meta = client.get_meta()
            ...

    Parameters:
        base_url: Root URL of the Anno server (e.g. ``http://localhost:8000``).
        api_key: Plaintext project API key (``ak_XXXXXXXX.yyyy...``).
        timeout: HTTP request timeout in seconds (default 30).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    # -- helpers -----------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> dict | list:
        """Issue an HTTP request and return the parsed JSON body.

        Raises:
            AnnoAPIError: On HTTP 4xx / 5xx.
            AnnoConnectionError: On network / timeout errors.
        """
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise AnnoConnectionError(str(exc)) from exc

        if response.status_code >= 400:
            detail = response.text
            raise AnnoAPIError(response.status_code, detail)

        # Some endpoints (e.g. image file) return non-JSON — handled by the
        # calling method, but for JSON endpoints we parse here.
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.content  # type: ignore[return-value]

    def _get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, *, json: dict | list) -> Any:
        return self._request("POST", path, json=json)

    def _patch(self, path: str, *, json: dict) -> Any:
        return self._request("PATCH", path, json=json)

    # -- project meta ------------------------------------------------------

    def get_meta(self) -> ProjectMeta:
        """Return project metadata including the label mapping."""
        data = self._get("/api/infers/project/meta")
        return ProjectMeta.from_dict(data)

    # -- images ------------------------------------------------------------

    def paginate_images(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        has_active_annotations: bool | None = None,
    ) -> PaginatedResponse[Image]:
        """List images in the project with offset/limit pagination.

        Parameters:
            limit: Page size (1–500, clamped by the server).
            offset: Number of images to skip.
            has_active_annotations:
                ``True`` — only images that have at least one active annotation.
                ``False`` — only images with zero active annotations.
                ``None`` — all images (default).
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if has_active_annotations is not None:
            params["has_active_annotations"] = str(has_active_annotations).lower()

        data = self._get("/api/infers/project/images", **params)
        return PaginatedResponse.from_dict(data, item_factory=Image.from_dict)

    def iter_images(
        self,
        *,
        limit: int = 100,
        has_active_annotations: bool | None = None,
    ) -> Iterator[Image]:
        """Yield every image in the project, auto-advancing offset.

        Parameters:
            limit: Number of images per HTTP request (page size, 1–500).
            has_active_annotations: Optional filter (see :meth:`paginate_images`).

        Yields:
            :class:`Image` instances one at a time.
        """
        offset = 0
        while True:
            page = self.paginate_images(
                limit=limit,
                offset=offset,
                has_active_annotations=has_active_annotations,
            )
            yield from page.items
            offset += limit
            if offset >= page.count:
                break

    def get_image(self, image_id: int) -> Image:
        """Get a single image by ID."""
        data = self._get(f"/api/infers/project/images/{image_id}")
        return Image.from_dict(data)

    def get_image_file(self, image_id: int) -> bytes:
        """Download the original image file bytes."""
        response = self._http.get(
            f"/api/infers/project/images/{image_id}/original_file"
        )
        if response.status_code >= 400:
            raise AnnoAPIError(response.status_code, response.text)
        return response.content

    # -- annotations -------------------------------------------------------

    def upload_annotations(
        self,
        image_id: int,
        annotations: list[Annotation],
    ) -> AnnotationBatchResult:
        """Submit a batch of annotations for a single image.

        Each annotation is processed independently by the backend — one
        failure does not affect the others.  Per-item errors are reported in
        the returned ``AnnotationBatchResult.results`` (with ``status="error"``)
        and are **not** raised as exceptions.

        Parameters:
            image_id: The image to annotate.
            annotations: List of :class:`Annotation` payloads.
        """
        body = {
            "annotations": [a.to_dict() for a in annotations],
        }
        data = self._post(
            f"/api/infers/project/images/{image_id}/annotations",
            json=body,
        )
        return AnnotationBatchResult.from_dict(data)

    def modify_annotation(
        self,
        image_id: int,
        annotation_id: int,
        annotation: Annotation,
    ) -> AnnotationModifyResult:
        """Modify an existing annotation (immutable pattern).

        The backend creates a **new** annotation row and deactivates the old
        one (``is_active=False``), recording an audit operation linking the two.

        Parameters:
            image_id: The image the annotation belongs to.
            annotation_id: The ID of the annotation to modify (must be active).
            annotation: The replacement annotation payload.
        """
        data = self._patch(
            f"/api/infers/project/images/{image_id}/annotations/{annotation_id}",
            json=annotation.to_dict(),
        )
        return AnnotationModifyResult.from_dict(data)
