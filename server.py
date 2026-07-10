"""Interactive SAM segmentation server for the Anno annotation platform.

This wraps HuggingFace ``transformers`` SAM (``facebook/sam-vit-base`` by
default) as an :class:`anno_sdk.InteractivePredictor`. The ``anno-sdk`` framework
supplies the whole web layer — session handshake, token auth, per-session image
caching, concurrency seats, and the FastAPI routes — so all we implement here is
the model logic:

    setup()          -> load the SAM checkpoint once per process
    embed_image()    -> run the image encoder once per session (cached as state)
    predict()        -> run the mask decoder per prompt against the cached embedding

Run it with ``python server.py`` (see the ``__main__`` block) or via the SDK CLI:

    anno-serve-interactive --predictor server:SamInteractivePredictor \\
        --auth-header X-API-Key --auth-header-value s3cr3t \\
        --public-url https://sam.example.com
"""

from __future__ import annotations

import io
import logging

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import SamModel, SamProcessor

from anno_sdk import (
    Annotation,
    InteractiveInferenceRequestMeta,
    InteractiveInferenceResponse,
    InteractivePredictor,
    Polygon2D,
)
from anno_sdk.prompts import (
    BoxPrompt,
    MaskPrompt,
    NegativePointPrompt,
    PositivePointPrompt,
    TextPrompt,
)

logger = logging.getLogger("sam_infer")

DEFAULT_MODEL_ID = "facebook/sam-vit-base"


def _pick_device() -> str:
    """Auto-detect the best available compute device: cuda -> mps -> cpu.

    Set the ``DEVICE`` env var to override auto-detection (e.g. ``DEVICE=cpu``).
    """
    import os

    if device := os.environ.get("DEVICE"):
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SamInteractivePredictor(InteractivePredictor):
    """Prompt-driven SAM predictor.

    The expensive image encoder runs once per session in :meth:`embed_image`; each
    prompt step in :meth:`predict` reuses the cached embedding and only runs the
    lightweight mask decoder.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self.device = _pick_device()
        self.model: SamModel | None = None
        self.processor: SamProcessor | None = None

    # -- lifecycle -----------------------------------------------------------

    def setup(self) -> None:
        """Load the SAM checkpoint + processor once, onto the chosen device."""
        logger.info("loading SAM model %s on %s", self.model_id, self.device)
        self.processor = SamProcessor.from_pretrained(self.model_id)
        self.model = SamModel.from_pretrained(self.model_id).to(self.device).eval()
        logger.info("SAM model ready")

    def embed_image(self, image_bytes: bytes, meta: dict | None):
        """Decode the upload and run SAM's image encoder once for the session.

        The returned state (PIL image + cached embedding) is handed to every
        :meth:`predict` call for this session.
        """
        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        return {"image": image, "embeddings": embeddings, "size": image.size}

    # -- inference -----------------------------------------------------------

    def predict(
        self, image_state, meta: InteractiveInferenceRequestMeta
    ) -> InteractiveInferenceResponse | None:
        """Segment against the cached embedding using the prompts in ``meta``.

        Returns a polygon candidate wrapped in an
        :class:`InteractiveInferenceResponse`, or ``None`` when the prompts yield
        no usable mask.
        """
        # Log the requested categories (label mapping) driving this prompt step.
        logger.info(
            "predict step image=%s session=%s step=%s: requested_types=%s label_mapping=%s prompts=%d",
            meta.image_id,
            meta.session_id,
            meta.step_index,
            meta.requested_types or "[]",
            meta.label_mapping or "{}",
            len(meta.prompts),
        )

        points: list[list[float]] = []
        labels: list[int] = []
        boxes: list[list[float]] = []

        for prompt in meta.prompts:
            if isinstance(prompt, PositivePointPrompt):
                points.append([prompt.x, prompt.y])
                labels.append(1)
            elif isinstance(prompt, NegativePointPrompt):
                points.append([prompt.x, prompt.y])
                labels.append(0)
            elif isinstance(prompt, BoxPrompt):
                boxes.append(
                    [prompt.x, prompt.y, prompt.x + prompt.width, prompt.y + prompt.height]
                )
            elif isinstance(prompt, (MaskPrompt, TextPrompt)):
                # v1: sam-vit-base has no text encoder, and low-res mask feedback
                # is not wired up yet. Extension point — skip for now.
                logger.debug("ignoring unsupported prompt type %s", prompt.prompt_type)

        if not points and not boxes:
            logger.info("no point/box prompts -> 0 candidates")
            return None

        image = image_state["image"]
        # The processor nests prompts per-image: points [[[x,y],...]],
        # labels [[l,...]], boxes [[[x0,y0,x1,y1]]]. Omit empty prompt groups.
        proc_kwargs: dict = {"images": image, "return_tensors": "pt"}
        if points:
            proc_kwargs["input_points"] = [points]
            proc_kwargs["input_labels"] = [labels]
        if boxes:
            proc_kwargs["input_boxes"] = [boxes]

        inputs = self.processor(**proc_kwargs).to(self.device)
        # We already have the image embedding — drop pixel_values so the encoder
        # is skipped and pass the cached embedding instead.
        inputs.pop("pixel_values", None)
        with torch.no_grad():
            outputs = self.model(
                image_embeddings=image_state["embeddings"],
                multimask_output=False,
                **inputs,
            )

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks[0] is (point_batch, num_masks, H, W) for the single image; we use
        # one point batch, so masks[0][0] is (num_masks, H, W). Pick the mask with
        # the highest predicted IoU (num_masks is 1 when multimask_output=False).
        mask_stack = masks[0][0]
        ious = outputs.iou_scores.cpu()[0, 0]
        best = int(ious.argmax())
        mask = mask_stack[best].numpy().astype(np.uint8)
        score = float(ious[best])

        # The category the candidate carries: SAM is class-agnostic, so we take the
        # single requested type (if any) as the candidate's label.
        candidate_label = meta.requested_types[0] if meta.requested_types else None
        # num_masks masks were produced; we return the single best one as the candidate.
        logger.info(
            "candidate produced: category=%s score=%.4f (masks evaluated=%d, candidates returned=1)",
            candidate_label,
            score,
            int(mask_stack.shape[0]),
        )

        try:
            geometry = Polygon2D.from_binary_mask(mask)
        except ValueError:
            # Empty / all-zero mask — no candidate for these prompts.
            logger.info("mask empty after threshold -> 0 candidates")
            return None

        return InteractiveInferenceResponse(
            annotation=Annotation.from_geometry(geometry),
            score=score,
            model_version=self.model_id,
        )


if __name__ == "__main__":
    import os

    from anno_sdk import InteractiveInferenceServer

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

    server = InteractiveInferenceServer(
        SamInteractivePredictor(os.environ.get("SAM_MODEL_ID", DEFAULT_MODEL_ID)),
        host=os.environ.get("SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("SERVER_PORT", "8422")),
        auth_header=os.environ.get("AUTH_HEADER", "X-API-Key"),
        auth_header_value=os.environ.get("PROVIDER_API_KEY", "changeme"),
        token_ttl_seconds=int(os.environ.get("TOKEN_TTL_SECONDS", "3600")),
        token_header=os.environ.get("TOKEN_HEADER", "X-Session-Token"),
        public_url=os.environ.get("PUBLIC_URL"),
        cors_origin=os.environ.get("CORS_ORIGIN"),
    )
    server.serve_forever()
