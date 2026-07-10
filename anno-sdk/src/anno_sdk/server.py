"""Reference inference server for anno-sdk.

Provides a FastAPI-based HTTP server that implements the Flow B wire protocol.
Install with ``pip install anno-sdk[server]`` to pull in FastAPI + uvicorn; the
base ``anno-sdk`` package does not require any web framework.

The user subclasses :class:`~anno_sdk.Predictor` and passes it (or a
``module:ClassName`` path) to ``anno-serve``::

    anno-serve --predictor my_package:MyPredictor --port 8080

Endpoints
---------
``GET /health`` — ``{"status": "ok"}``
``POST /predict`` — multipart/form-data (``image`` + ``metadata`` JSON) → JSON

Features
--------
* Auto-generated OpenAPI docs at ``/docs`` (Swagger) and ``/redoc``.
* Optional service-side auth via ``--auth-header`` / ``--auth-query``,
  mirroring ``InferenceServiceProvider.auth_*``.
* ``/ready`` endpoint (checks predictor has been ``setup()`` successfully).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Annotated, Any

from .handler import Predictor

logger = logging.getLogger("anno_sdk.server")


# ---------------------------------------------------------------------------
# _fastapi_app — lazy-built when InferenceServer starts (avoids import cost
# for users who only use the SDK client).
# ---------------------------------------------------------------------------

_FASTAPI_AVAILABLE = False
try:
    from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile  # noqa: F811
    from fastapi.responses import JSONResponse
    from starlette.requests import Request  # injected by FastAPI Depends

    import uvicorn

    _FASTAPI_AVAILABLE = True
except ImportError:
    Request = None  # type: ignore[assignment,misc]


def _require_server_extras() -> None:
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "anno-sdk server extras not installed.  Run:\n"
            "    pip install anno-sdk[server]"
        )


ResponseDict = dict[str, Any]
"""Type alias for the JSON-serializable dict the server returns."""


# ---------------------------------------------------------------------------
# Auth dependencies (FastAPI Depends)
# ---------------------------------------------------------------------------


def _make_auth_dep(
    auth_header: str | None,
    auth_header_value: str | None,
    auth_query: str | None,
    auth_query_value: str | None,
):
    """Return a single FastAPI dependency that enforces all configured auth.

    Returns ``None`` if no auth is configured (all requests accepted).
    """
    if not auth_header and not auth_query:
        return None

    _hdr_name = auth_header
    _hdr_val = auth_header_value
    _qry_name = auth_query
    _qry_val = auth_query_value

    async def _check(request: Request) -> None:  # type: ignore[name-defined]
        if _hdr_name is not None and _hdr_val is not None:
            if request.headers.get(_hdr_name) != _hdr_val:
                raise HTTPException(status_code=401, detail="unauthorized")
        if _qry_name is not None and _qry_val is not None:
            if request.query_params.get(_qry_name) != _qry_val:
                raise HTTPException(status_code=401, detail="unauthorized")

    return Depends(_check)


# ---------------------------------------------------------------------------
# InferenceServer
# ---------------------------------------------------------------------------


class InferenceServer:
    """A FastAPI inference HTTP server wrapping a :class:`~anno_sdk.Predictor`.

    Parameters
    ----------
    predictor:
        An instance of a :class:`~anno_sdk.Predictor` subclass.
    host:
        Address to bind (default ``"0.0.0.0"``).
    port:
        Port to listen on (default ``8080``).
    auth_header / auth_header_value:
        If set, require an HTTP header with the given value on ``POST /predict``.
        Mirrors ``InferenceServiceProvider.auth_type="header"``.
    auth_query / auth_query_value:
        If set, require a query parameter with the given value.
        Mirrors ``InferenceServiceProvider.auth_type="query"``.

    Example::

        from anno_sdk import InferenceServer, Predictor, Annotation, Box2D

        class MyModel(Predictor):
            def predict(self, image_bytes, meta):
                return [Annotation.from_geometry(Box2D(0, 0, 10, 10))]

        server = InferenceServer(MyModel(), port=8080)
        server.serve_forever()
    """

    def __init__(
        self,
        predictor: Predictor,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        auth_header: str | None = None,
        auth_header_value: str | None = None,
        auth_query: str | None = None,
        auth_query_value: str | None = None,
    ) -> None:
        _require_server_extras()

        self.host = host
        self.port = port
        self.predictor = predictor
        self._auth_dep = _make_auth_dep(
            auth_header, auth_header_value, auth_query, auth_query_value
        )

        self._app = FastAPI(
            title="anno-sdk inference service",
            version="0.2.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        self._register_routes()

    # -- route registration ---------------------------------------------------

    def _register_routes(self) -> None:
        app = self._app
        predictor = self.predictor
        auth_dep = self._auth_dep

        @app.get("/health")
        async def health() -> ResponseDict:
            return {"status": "ok"}

        @app.get("/ready")
        async def ready() -> ResponseDict:
            try:
                predictor.setup()  # noop if already done
            except Exception as exc:
                return JSONResponse(
                    status_code=503,
                    content={"status": "not ready", "error": str(exc)},
                )
            return {"status": "ready"}

        # Build /predict with or without auth dep.
        predict_kwargs: dict[str, Any] = {}
        if auth_dep is not None:
            predict_kwargs["dependencies"] = [auth_dep]

        @app.post("/predict", **predict_kwargs)
        async def predict(
            image: UploadFile,
            metadata: Annotated[str, Form()],
        ) -> ResponseDict:
            image_bytes = await image.read()
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"invalid metadata JSON: {exc}")
            try:
                return predictor.serve(image_bytes, meta_dict)
            except Exception as exc:
                logger.exception("prediction failed for image %s", meta_dict.get("image_id"))
                raise HTTPException(status_code=500, detail=f"prediction failed: {exc}")

    # -- server lifecycle -----------------------------------------------------

    def serve_forever(self) -> None:
        """Block and serve requests via uvicorn until interrupted."""
        logger.info(
            "anno-serve listening on http://%s:%d, auth=%s, docs=http://%s:%d/docs",
            self.host, self.port,
            "on" if self._auth_dep else "off",
            self.host, self.port,
        )
        uvicorn.run(self._app, host=self.host, port=self.port, log_config=None)

    def shutdown(self) -> None:
        """No-op (uvicorn handles shutdown via signal handlers)."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app(predictor: Predictor, **kwargs) -> InferenceServer:
    """Create an :class:`InferenceServer` instance.

    Convenience alias for ``InferenceServer(predictor, **kwargs)``.
    """
    return InferenceServer(predictor, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _import_predictor(spec: str) -> Predictor:
    """Resolve ``module:ClassName`` or ``module:ClassName()`` to a Predictor instance."""
    if ":" not in spec:
        raise ValueError(f"Expected 'module:ClassName', got {spec!r}")

    module_path, class_name = spec.split(":", 1)
    call = class_name.endswith("()")
    if call:
        class_name = class_name[:-2]

    __import__(module_path)
    module = sys.modules[module_path]
    cls = getattr(module, class_name)

    if call:
        instance = cls()
    elif isinstance(cls, type) and issubclass(cls, Predictor):
        instance = cls()
    elif isinstance(cls, Predictor):
        instance = cls
    else:
        raise TypeError(f"{class_name!r} is not a Predictor subclass or instance")

    if not isinstance(instance, Predictor):
        raise TypeError(f"{instance!r} is not a Predictor instance")
    return instance


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="anno-sdk inference server")
    p.add_argument("--predictor", required=True, help="module:ClassName (e.g. my_app:MyModel)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--auth-header", default=None, help="Require this header on /predict")
    p.add_argument(
        "--auth-header-value", default=None, metavar="VALUE",
        help="Expected header value (requires --auth-header)",
    )
    p.add_argument(
        "--auth-query", default=None, help="Require this query param on /predict",
    )
    p.add_argument(
        "--auth-query-value", default=None, metavar="VALUE",
        help="Expected query value (requires --auth-query)",
    )

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.auth_header and not args.auth_header_value:
        p.error("--auth-header-value is required when --auth-header is set")
    if args.auth_query and not args.auth_query_value:
        p.error("--auth-query-value is required when --auth-query is set")

    _require_server_extras()
    predictor = _import_predictor(args.predictor)
    server = InferenceServer(
        predictor,
        host=args.host,
        port=args.port,
        auth_header=args.auth_header,
        auth_header_value=args.auth_header_value,
        auth_query=args.auth_query,
        auth_query_value=args.auth_query_value,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
