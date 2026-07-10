"""The :class:`InteractivePredictor` contract — model logic for interactive inference.

This is the *only* thing a service author implements. It is deliberately kept in
its own module with **no web-framework dependency**, so it can be subclassed and
unit-tested anywhere without installing ``anno-sdk[server]``. The FastAPI wiring
that drives it lives in :mod:`anno_sdk.interactive_server`.
"""

from __future__ import annotations

from .interactive import InteractiveInferenceRequestMeta, InteractiveInferenceResponse
from .types import Annotation


class InteractivePredictor:
    """Model logic for interactive (prompt-driven) inference.

    Lifecycle per session::

        setup()                      # once per process, before first use
        embed_image(bytes, meta)     # once per session, on /infer_image
        predict(state, meta)         # per prompt, on /predict  (override this)

    Only :meth:`predict` is required. Override :meth:`embed_image` to precompute
    an embedding (its return value is cached as ``state`` and handed to every
    :meth:`predict`); by default the raw image bytes are cached. Override
    :meth:`setup` to load model weights once.
    """

    def setup(self) -> None:
        """One-time initialization (e.g. load weights). Runs once, lazily."""

    def embed_image(self, image_bytes: bytes, meta: dict | None):
        """Turn the uploaded image into the cached per-session ``state``.

        Runs once per session (on ``/infer_image``). Return whatever
        :meth:`predict` needs — an embedding tensor, a wrapper object, or (by
        default) the raw bytes. ``meta`` is the optional JSON block the frontend
        sent alongside the image, or ``None``.
        """
        return image_bytes

    def predict(
        self, image_state, meta: InteractiveInferenceRequestMeta
    ) -> InteractiveInferenceResponse | Annotation | None:
        """Run the prompts in ``meta`` against the cached ``image_state``.

        ``meta.prompts`` is a list of typed :data:`~anno_sdk.prompts.Prompt`
        objects (``BoxPrompt``, ``PositivePointPrompt``, ``NegativePointPrompt``,
        ``MaskPrompt``, ``TextPrompt``) — dispatch on their type directly.

        **Must be overridden.** Return an :class:`InteractiveInferenceResponse`,
        or a bare :class:`~anno_sdk.types.Annotation` (wrapped automatically), or
        ``None`` when the prompts yield no candidate.
        """
        raise NotImplementedError("InteractivePredictor subclasses must implement predict()")

    # -- internals -----------------------------------------------------------

    def _ensure_setup(self) -> None:
        if not getattr(self, "_is_setup", False):
            self.setup()
            self._is_setup = True

    def embed(self, image_bytes: bytes, meta: dict | None):
        self._ensure_setup()
        return self.embed_image(image_bytes, meta)

    def serve_predict(self, image_state, metadata: dict) -> dict:
        """Run :meth:`predict` for one prompt step and return the wire body."""
        self._ensure_setup()
        meta = InteractiveInferenceRequestMeta.from_dict(metadata)
        out = self.predict(image_state, meta)
        return _normalize_candidate(out).to_dict()


def _normalize_candidate(out) -> InteractiveInferenceResponse:
    if out is None:
        return InteractiveInferenceResponse()
    if isinstance(out, InteractiveInferenceResponse):
        return out
    if isinstance(out, Annotation):
        return InteractiveInferenceResponse(annotation=out)
    raise TypeError(
        "predict() must return InteractiveInferenceResponse, Annotation, or None; "
        f"got {type(out).__name__}"
    )
