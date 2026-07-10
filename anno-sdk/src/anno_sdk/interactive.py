"""Contract for *interactive* inference (SAM / SAM2 / MedSAM and similar).

Interactive inference models do not annotate a whole image in one shot; a user
iteratively supplies prompts (a box, positive or negative points, a mask, or
text) and the model returns a single refined candidate each step.

The interaction is **frontend-direct**: the browser issues one HTTP call per
step straight to the service, so the latency-sensitive loop never passes through
the Anno server. To avoid handing the service's long-term credential to the
browser, the Anno server first performs a short server→service handshake — it
POSTs a :class:`InteractiveSessionCreateRequest` (authenticated with the
provider credential) and the service mints a **short-lived token** returned in
:class:`InteractiveSessionCreateResponse`. The server relays that token to the
frontend, which presents it on each direct call; the service validates its own
token. When the user confirms, the frontend sends the final candidate back to
the Anno server (polygon only), which persists it.

Per-step wire shapes (frontend → service) reuse
:class:`InteractiveInferenceRequestMeta` / :class:`InteractiveInferenceResponse`.
As with the batch :mod:`~anno_sdk.inference` contract, the image bytes travel as
a separate multipart part (``image``); this module only models the JSON parts.

Prompt types
------------
``box``, ``positive_point``, ``negative_point``, ``mask``, ``text``. On the wire
prompts are a list of plain dicts, each tagged with ``"type"``; the remaining keys
are prompt-specific (e.g. a box's ``x/y/width/height``, a point's ``x/y``, a
mask's ``points``/RLE, text's ``text``). :class:`InteractiveInferenceRequestMeta`
parses them into the typed :data:`~anno_sdk.prompts.Prompt` objects defined in
:mod:`anno_sdk.prompts`, so a predictor iterates concrete objects while the wire
format stays dict-based (and backend-compatible).

Wire shapes
-----------
Request ``metadata`` part::

    {
        "image_id": 42,
        "session_id": 7,
        "step_index": 3,
        "prompts": [
            {"type": "box", "x": 10, "y": 20, "width": 100, "height": 50},
            {"type": "positive_point", "x": 55, "y": 40},
        ],
        "label_mapping": {"cat": 0, "dog": 1},
        "requested_types": ["polygon"],
        "width": 1920,
        "height": 1080,
        "client_ref": null
    }

Response body::

    {
        "annotation": <Annotation.to_dict()> | null,
        "score": 0.97,
        "model_version": "sam2-hiera-large",
        "raw": { ... }
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .prompts import (
    PROMPT_BOX,
    PROMPT_MASK,
    PROMPT_NEGATIVE_POINT,
    PROMPT_POSITIVE_POINT,
    PROMPT_TEXT,
    PROMPT_TYPES,
    Prompt,
    parse_prompts,
)
from .types import Annotation

# Re-export the prompt-type constants (their canonical home is anno_sdk.prompts)
# so existing ``from anno_sdk.interactive import PROMPT_BOX`` imports keep working.
__all__ = [
    "IMAGE_PART_NAME",
    "METADATA_PART_NAME",
    "PROMPT_BOX",
    "PROMPT_POSITIVE_POINT",
    "PROMPT_NEGATIVE_POINT",
    "PROMPT_MASK",
    "PROMPT_TEXT",
    "PROMPT_TYPES",
    "InteractiveInferenceRequestMeta",
    "InteractiveInferenceResponse",
    "InteractiveSessionCreateRequest",
    "InteractiveSessionCreateResponse",
]

#: Name of the multipart part carrying the raw image bytes.
IMAGE_PART_NAME = "image"
#: Name of the multipart part carrying the JSON metadata block.
METADATA_PART_NAME = "metadata"


@dataclass
class InteractiveInferenceRequestMeta:
    """The JSON metadata block accompanying the image bytes for one step.

    The image itself is sent as binary in a separate multipart part, so it is
    deliberately *not* a field here.
    """

    image_id: int
    session_id: int
    step_index: int
    prompts: list[Prompt]
    label_mapping: dict = field(default_factory=dict)
    requested_types: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    client_ref: str | None = None

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "session_id": self.session_id,
            "step_index": self.step_index,
            "prompts": [p.to_dict() for p in self.prompts],
            "label_mapping": self.label_mapping,
            "requested_types": list(self.requested_types),
            "width": self.width,
            "height": self.height,
            "client_ref": self.client_ref,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InteractiveInferenceRequestMeta:
        return cls(
            image_id=data["image_id"],
            session_id=data["session_id"],
            step_index=data["step_index"],
            prompts=parse_prompts(data.get("prompts") or []),
            label_mapping=data.get("label_mapping") or {},
            requested_types=list(data.get("requested_types") or []),
            width=data.get("width"),
            height=data.get("height"),
            client_ref=data.get("client_ref"),
        )


@dataclass
class InteractiveInferenceResponse:
    """The candidate an interactive service returns for a single step.

    Reuses :class:`~anno_sdk.types.Annotation` for the geometry, so a committed
    candidate is byte-identical to what the annotation endpoints already accept.
    ``annotation`` may be ``None`` when the model produced no candidate for the
    given prompts. ``score`` is the model's confidence, ``raw`` an optional
    passthrough of the service's raw output.
    """

    annotation: Annotation | None = None
    score: float | None = None
    model_version: str | None = None
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "annotation": self.annotation.to_dict() if self.annotation is not None else None,
            "score": self.score,
            "model_version": self.model_version,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InteractiveInferenceResponse:
        ann = data.get("annotation")
        return cls(
            annotation=Annotation.from_dict(ann) if ann else None,
            score=data.get("score"),
            model_version=data.get("model_version"),
            raw=data.get("raw") or {},
        )


@dataclass
class InteractiveSessionCreateRequest:
    """Server → service handshake opening an interactive session.

    The Anno server sends this once (authenticated with the provider credential)
    to obtain a short-lived token the frontend then uses for its direct per-step
    calls. Carries enough context for the service to associate or pre-warm the
    session (e.g. compute an image embedding up front); it deliberately does not
    include image bytes — the frontend uploads those on the per-step calls.
    """

    session_id: int
    image_id: int
    requested_types: list[str] = field(default_factory=list)
    label_mapping: dict = field(default_factory=dict)
    ttl_seconds: int | None = None
    width: int | None = None
    height: int | None = None
    client_ref: str | None = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "image_id": self.image_id,
            "requested_types": list(self.requested_types),
            "label_mapping": self.label_mapping,
            "ttl_seconds": self.ttl_seconds,
            "width": self.width,
            "height": self.height,
            "client_ref": self.client_ref,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InteractiveSessionCreateRequest:
        return cls(
            session_id=data["session_id"],
            image_id=data["image_id"],
            requested_types=list(data.get("requested_types") or []),
            label_mapping=data.get("label_mapping") or {},
            ttl_seconds=data.get("ttl_seconds"),
            width=data.get("width"),
            height=data.get("height"),
            client_ref=data.get("client_ref"),
        )


@dataclass
class InteractiveSessionCreateResponse:
    """Service → server reply to :class:`InteractiveSessionCreateRequest`.

    ``token`` is the short-lived credential the frontend presents on its direct
    calls; the service validates it itself. ``expires_at`` is an optional ISO-8601
    timestamp (the server may otherwise derive expiry from the requested TTL).
    ``session_ref`` is an optional service-side handle, and ``predict_url`` an
    optional per-session endpoint override (else the provider's configured URL).
    """

    token: str
    expires_at: str | None = None
    session_ref: str | None = None
    predict_url: str | None = None
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "expires_at": self.expires_at,
            "session_ref": self.session_ref,
            "predict_url": self.predict_url,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InteractiveSessionCreateResponse:
        return cls(
            token=data["token"],
            expires_at=data.get("expires_at"),
            session_ref=data.get("session_ref"),
            predict_url=data.get("predict_url"),
            raw=data.get("raw") or {},
        )
