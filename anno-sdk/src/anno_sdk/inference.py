"""Server <-> inference-service contract for server-driven auto-annotation.

In the server-initiated flow (anno_infers "Flow B"), the Anno server runs a
background job that, for each image, POSTs the raw image bytes plus a small
JSON metadata block to a registered inference service, and expects the service
to return annotations in the standard wire format.

The image bytes travel as a separate multipart part (``image``); this module
only models the JSON ``metadata`` part of the request and the JSON response
body. Both sides import these types so there is exactly one canonical shape.

Wire shapes
-----------
Request ``metadata`` part::

    {
        "image_id": 42,
        "task_id": 7,
        "label_mapping": {"cat": 0, "dog": 1},
        "requested_types": ["box", "polygon"],
        "width": 1920,
        "height": 1080,
        "client_ref": null
    }

Response body::

    {
        "annotations": [ <Annotation.to_dict()>, ... ],
        "model_version": "yolo-v8.1"
    }
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Annotation, _annotation_from_dict

#: Name of the multipart part carrying the raw image bytes.
IMAGE_PART_NAME = "image"
#: Name of the multipart part carrying the JSON metadata block.
METADATA_PART_NAME = "metadata"


@dataclass
class InferenceRequestMeta:
    """The JSON metadata block accompanying the image bytes in a request.

    The image itself is sent as binary in a separate multipart part, so it is
    deliberately *not* a field here.
    """

    image_id: int
    task_id: int
    label_mapping: dict
    requested_types: list[str]
    width: int | None = None
    height: int | None = None
    client_ref: str | None = None

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "task_id": self.task_id,
            "label_mapping": self.label_mapping,
            "requested_types": list(self.requested_types),
            "width": self.width,
            "height": self.height,
            "client_ref": self.client_ref,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InferenceRequestMeta:
        return cls(
            image_id=data["image_id"],
            task_id=data["task_id"],
            label_mapping=data.get("label_mapping") or {},
            requested_types=list(data.get("requested_types") or []),
            width=data.get("width"),
            height=data.get("height"),
            client_ref=data.get("client_ref"),
        )


@dataclass
class InferenceResponse:
    """The response an inference service returns for a single image.

    Reuses :class:`~anno_sdk.types.Annotation`, so each item is byte-identical
    to what the annotation-submission endpoint already accepts.
    """

    annotations: list[Annotation]
    model_version: str | None = None

    def to_dict(self) -> dict:
        return {
            "annotations": [a.to_dict() for a in self.annotations],
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InferenceResponse:
        return cls(
            annotations=[_annotation_from_dict(a) for a in data.get("annotations") or []],
            model_version=data.get("model_version"),
        )
