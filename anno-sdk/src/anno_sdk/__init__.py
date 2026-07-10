"""Anno SDK — Python client for the Anno annotation platform inference API.

Usage::

    from anno_sdk import Client, Annotation, Box2D, Polygon2D

    client = Client(base_url="http://localhost:8000", api_key="ak_...")
    meta = client.get_meta()

    # Upload a box annotation
    ann = Annotation(label=1, geometry=Box2D(10, 20, 100, 50))
    result = client.upload_annotations(image_id=42, annotations=[ann])

    # Iterate over all images
    for img in client.iter_images():
        print(img.file_name)
"""

from .client import Client
from .exceptions import AnnoAPIError, AnnoConnectionError, AnnoSDKError
from .handler import PredictFn, Predictor, serve_predict

try:
    from .server import InferenceServer, create_app
except ImportError:  # pragma: no cover — server extras not installed
    InferenceServer = None  # type: ignore[assignment,misc]
    create_app = None  # type: ignore[assignment,misc]
# InteractivePredictor + server: predictor base has no web-framework deps
# (subclass it anywhere); constructing the server requires anno-sdk[server].
from .inference import InferenceRequestMeta, InferenceResponse
from .interactive import (
    PROMPT_TYPES,
    InteractiveInferenceRequestMeta,
    InteractiveInferenceResponse,
    InteractiveSessionCreateRequest,
    InteractiveSessionCreateResponse,
)
from .interactive_predictor import InteractivePredictor
from .interactive_server import (
    InteractiveInferenceServer,
    SessionStore,
    create_interactive_app,
)
from .prompts import (
    BoxPrompt,
    MaskPrompt,
    NegativePointPrompt,
    PositivePointPrompt,
    Prompt,
    TextPrompt,
    parse_prompt,
    parse_prompts,
)
from .types import (
    Annotation,
    AnnotationBatchResult,
    AnnotationModifyResult,
    AnnotationResultItem,
    Box2D,
    GeometryDO,
    Image,
    Keypoint2D,
    PaginatedResponse,
    Polygon2D,
    ProjectMeta,
    RotatedBox2D,
)

__all__ = [
    "Client",
    # Geometry
    "Box2D",
    "RotatedBox2D",
    "Polygon2D",
    "Keypoint2D",
    "GeometryDO",
    # Payload
    "Annotation",
    # Responses / results
    "Image",
    "ProjectMeta",
    "PaginatedResponse",
    "AnnotationBatchResult",
    "AnnotationResultItem",
    "AnnotationModifyResult",
    # Server-driven inference contract
    "InferenceRequestMeta",
    "InferenceResponse",
    # Interactive inference contract
    "InteractiveInferenceRequestMeta",
    "InteractiveInferenceResponse",
    "InteractiveSessionCreateRequest",
    "InteractiveSessionCreateResponse",
    "PROMPT_TYPES",
    # Typed interactive prompts
    "Prompt",
    "BoxPrompt",
    "PositivePointPrompt",
    "NegativePointPrompt",
    "MaskPrompt",
    "TextPrompt",
    "parse_prompt",
    "parse_prompts",
    # Interactive server (base class dep-free; server needs anno-sdk[server])
    "InteractivePredictor",
    "InteractiveInferenceServer",
    "create_interactive_app",
    "SessionStore",
    "Predictor",
    "serve_predict",
    "PredictFn",
    # Server (available when installed with anno-sdk[server])
    "InferenceServer",
    "create_app",
    # Exceptions
    "AnnoSDKError",
    "AnnoAPIError",
    "AnnoConnectionError",
]
