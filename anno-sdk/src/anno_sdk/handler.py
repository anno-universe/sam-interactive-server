"""Server-side helper for inference-service authors.

An inference service only needs to (1) receive the image bytes + metadata, and
(2) return annotations. It does **not** need a :class:`~anno_sdk.client.Client`,
the Anno server URL, or a project API key — everything required arrives in the
request.

Two equivalent entry points are provided:

**Class-based** (preferred when the model has state to load once, or you want
pre/post-processing hooks). Subclass :class:`Predictor` and override
``predict``::

    from anno_sdk import Annotation, Box2D, Predictor

    class MyModel(Predictor):
        def setup(self):
            self.model = load_weights()          # runs once, before first predict

        def predict(self, image_bytes, meta):
            # run self.model on image_bytes ...
            return [Annotation.from_geometry(Box2D(0, 0, 10, 10), label=1)]

    predictor = MyModel()

    # FastAPI example
    @app.post("/predict")
    async def endpoint(image: UploadFile, metadata: str = Form(...)):
        import json
        return predictor.serve(await image.read(), json.loads(metadata))

**Function-based** (thin wrapper over a stateless ``predict`` callable)::

    from anno_sdk import Annotation, Box2D, serve_predict

    def predict(image_bytes, meta):
        return [Annotation.from_geometry(Box2D(0, 0, 10, 10), label=1)]

    @app.post("/predict")
    async def endpoint(image: UploadFile, metadata: str = Form(...)):
        import json
        return serve_predict(await image.read(), json.loads(metadata), predict)
"""

from __future__ import annotations

from collections.abc import Callable

from .inference import InferenceRequestMeta, InferenceResponse
from .types import Annotation

#: Signature of a user-supplied prediction function.
PredictFn = Callable[[bytes, InferenceRequestMeta], list[Annotation]]


class Predictor:
    """Base class for an inference service. Subclass and override :meth:`predict`.

    The request lifecycle for each image, driven by :meth:`serve`, is::

        setup()        # once, lazily, before the first prediction
        preprocess()   # raw bytes -> model inputs
        predict()      # model inputs -> annotations   (override this)
        postprocess()  # annotations -> annotations

    Only :meth:`predict` is required; the other three default to no-ops so a
    minimal subclass is just one method. Override :meth:`setup` to load model
    weights once, and :meth:`preprocess` / :meth:`postprocess` to wrap the model
    call.
    """

    def setup(self) -> None:
        """One-time initialization, e.g. loading model weights.

        Called automatically before the first prediction (and only once).
        Default implementation does nothing.
        """

    def preprocess(self, image_bytes: bytes, meta: InferenceRequestMeta):
        """Transform raw image bytes into the inputs :meth:`predict` expects.

        Default implementation passes the bytes through unchanged.
        """
        return image_bytes

    def predict(self, inputs, meta: InferenceRequestMeta) -> list[Annotation]:
        """Run the model and return annotations. **Must be overridden.**

        ``inputs`` is whatever :meth:`preprocess` returned (the raw image bytes
        by default).
        """
        raise NotImplementedError("Predictor subclasses must implement predict()")

    def postprocess(
        self, annotations: list[Annotation], meta: InferenceRequestMeta
    ) -> list[Annotation]:
        """Adjust annotations after :meth:`predict` (filter, clip, relabel, …).

        Default implementation passes them through unchanged.
        """
        return annotations

    def serve(self, image_bytes: bytes, metadata: dict) -> dict:
        """Parse a request, run the pipeline, and return the response body.

        Parameters
        ----------
        image_bytes:
            Raw bytes of the image to annotate (the ``image`` multipart part).
        metadata:
            The decoded JSON ``metadata`` part (see :class:`InferenceRequestMeta`).

        Returns
        -------
        dict
            The :class:`InferenceResponse` wire body to return to the Anno server.
        """
        self._ensure_setup()
        meta = InferenceRequestMeta.from_dict(metadata)
        inputs = self.preprocess(image_bytes, meta)
        annotations = self.predict(inputs, meta)
        annotations = self.postprocess(list(annotations), meta)
        return InferenceResponse(annotations=list(annotations)).to_dict()

    def _ensure_setup(self) -> None:
        """Call :meth:`setup` exactly once, on the first :meth:`serve`.

        Uses ``getattr`` for the flag so subclasses that skip
        ``super().__init__()`` still work.
        """
        if not getattr(self, "_is_setup", False):
            self.setup()
            self._is_setup = True


class _FunctionPredictor(Predictor):
    """Adapter that exposes a bare ``predict`` callable as a :class:`Predictor`."""

    def __init__(self, fn: PredictFn) -> None:
        self._fn = fn

    def predict(self, inputs, meta: InferenceRequestMeta) -> list[Annotation]:
        return self._fn(inputs, meta)


def serve_predict(image_bytes: bytes, metadata: dict, predict: PredictFn) -> dict:
    """Parse a request, run ``predict``, and return the JSON-serializable response.

    Thin functional wrapper over :class:`Predictor` for stateless prediction
    callables.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the image to annotate (the ``image`` multipart part).
    metadata:
        The decoded JSON ``metadata`` part (see :class:`InferenceRequestMeta`).
    predict:
        Callable that runs the model and returns a list of
        :class:`~anno_sdk.types.Annotation`.

    Returns
    -------
    dict
        The :class:`InferenceResponse` wire body to return to the Anno server.
    """
    return _FunctionPredictor(predict).serve(image_bytes, metadata)
