"""Reference *interactive* inference server for anno-sdk.

A batteries-included FastAPI server that implements the interactive auth +
image-caching protocol so a service author only writes the model logic. Install
with ``pip install anno-sdk[server]``.

Protocol
--------
Three roles, two credentials:

``POST /session`` — called **by the Anno server** (server→service handshake),
    guarded by the long-term *provider credential* (``--auth-header`` /
    ``--auth-query``, mirroring ``InteractiveInferenceServiceProvider.auth_*``).
    Body is an :class:`~anno_sdk.InteractiveSessionCreateRequest`. The server
    mints a **short-lived token** bound to the session and returns an
    :class:`~anno_sdk.InteractiveSessionCreateResponse` (``token``,
    ``expires_at``, ``session_ref``, ``predict_url``). The Anno server relays the
    token + ``predict_url`` to the browser.

``POST /{session_id}/infer_image`` — called **by the browser once**, guarded by
    the session token. Uploads the image (multipart ``image``); the server runs
    :meth:`InteractivePredictor.embed_image` a single time so subsequent prompts
    are cheap. This is the whole point of the session: the frontend does not
    re-upload the image on every prompt.

``POST /{session_id}/predict`` — called **by the browser per prompt**, guarded by
    the session token. Body is an :class:`~anno_sdk.InteractiveInferenceRequestMeta`
    JSON (**prompts only, no image**). Runs :meth:`InteractivePredictor.predict`
    against the cached image state and returns an
    :class:`~anno_sdk.InteractiveInferenceResponse`.

``POST /session/{session_id}/complete`` — called **by the Anno server**, guarded
    by the *provider credential* (same as ``/session``). Evicts the session's
    cached image once the session is committed / discarded.

``DELETE /{session_id}`` — optional; the browser may evict its cached image
    early.

The service author subclasses :class:`InteractivePredictor` (override
``embed_image`` to precompute an embedding, ``predict`` to run prompts) and runs::

    anno-serve-interactive --predictor my_pkg:MyModel \\
        --auth-header X-API-Key --auth-header-value s3cr3t \\
        --public-url https://sam.example.com
"""

# NOTE: intentionally no ``from __future__ import annotations`` — the route
# handlers use ``Header(alias=token_header)`` with a closure variable, which
# FastAPI can only bind when annotations are real objects (not lazy strings).

import argparse
import json
import logging
import secrets
import sys
import threading
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from .interactive import (
    InteractiveSessionCreateRequest,
    InteractiveSessionCreateResponse,
)
from .interactive_predictor import InteractivePredictor
from .server import _make_auth_dep, _require_server_extras

logger = logging.getLogger("anno_sdk.interactive_server")

_FASTAPI_AVAILABLE = False
try:
    import uvicorn
    from fastapi import FastAPI, Form, Header, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from starlette.requests import Request  # noqa: F401 — used via _make_auth_dep

    _FASTAPI_AVAILABLE = True
except ImportError:
    pass


ResponseDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Session store — token + cached image state, keyed by session_id
# ---------------------------------------------------------------------------


class _Session:
    __slots__ = ("token", "expires_at", "image_state", "has_image", "last_active")

    def __init__(self, token: str, expires_at: datetime, now: datetime) -> None:
        self.token = token
        self.expires_at = expires_at
        self.image_state: Any = None
        self.has_image = False
        self.last_active = now


class SessionStore:
    """In-memory session store (token + cached image state).

    Thread-safe for a single process. For multi-worker / multi-replica
    deployments, subclass and override these methods to share state (Redis, a DB,
    etc.); the server calls only these.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def create(self, session_id: str, ttl_seconds: int) -> tuple[str, datetime]:
        token = secrets.token_urlsafe(32)
        now = self._now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        with self._lock:
            self._sessions[session_id] = _Session(token, expires_at, now)
        return token, expires_at

    def authenticate(self, session_id: str, token: str) -> _Session | None:
        """Return the live session iff ``token`` matches and has not expired."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None or not token:
                return None
            now = self._now()
            if sess.expires_at <= now:
                self._sessions.pop(session_id, None)
                return None
            if not secrets.compare_digest(sess.token, token):
                return None
            sess.last_active = now
            return sess

    def set_image(self, session_id: str, image_state: Any) -> None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                sess.image_state = image_state
                sess.has_image = True
                sess.last_active = self._now()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# InteractiveInferenceServer
# ---------------------------------------------------------------------------


class InteractiveInferenceServer:
    """FastAPI server implementing the interactive protocol around a predictor.

    Parameters
    ----------
    predictor:
        An :class:`InteractivePredictor` instance.
    host / port:
        Bind address / port.
    auth_header / auth_header_value / auth_query / auth_query_value:
        The *provider credential* guarding ``POST /session`` — mirrors
        ``InteractiveInferenceServiceProvider.auth_*`` so only the Anno server
        (which holds the secret) can open sessions.
    token_ttl_seconds:
        Fallback lifetime of a minted token when the request omits ``ttl_seconds``.
    token_header:
        Header the browser presents on ``/infer_image`` and ``/predict``
        (default ``"X-Session-Token"``).
    public_url:
        The server's browser-reachable base URL, returned as ``predict_url`` in
        the handshake so the Anno server can relay it to the frontend. The
        frontend then calls ``{predict_url}/{session_id}/infer_image`` and
        ``{predict_url}/{session_id}/predict``.
    store:
        A :class:`SessionStore` (or subclass for shared state). Defaults to an
        in-memory store.
    """

    def __init__(
        self,
        predictor: InteractivePredictor,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        auth_header: str | None = None,
        auth_header_value: str | None = None,
        auth_query: str | None = None,
        auth_query_value: str | None = None,
        token_ttl_seconds: int = 3600,
        token_header: str = "X-Session-Token",
        public_url: str | None = None,
        cors_origin: str | None = None,
        store: SessionStore | None = None,
    ) -> None:
        _require_server_extras()

        self.host = host
        self.port = port
        self.predictor = predictor
        self.token_ttl_seconds = token_ttl_seconds
        self.token_header = token_header
        self.public_url = public_url.rstrip("/") if public_url else None
        self.store = store or SessionStore()
        self._session_auth_dep = _make_auth_dep(
            auth_header, auth_header_value, auth_query, auth_query_value
        )

        self._app = FastAPI(
            title="anno-sdk interactive inference service",
            version="0.4.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        if cors_origin:
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=[cors_origin],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        self._register_routes()

    # -- route registration ---------------------------------------------------

    def _register_routes(self) -> None:
        app = self._app
        predictor = self.predictor
        store = self.store
        token_header = self.token_header

        def _authed_session(session_id: str, token: str | None) -> _Session:
            sess = store.authenticate(session_id, token or "")
            if sess is None:
                raise HTTPException(status_code=401, detail="invalid or expired session token")
            return sess

        @app.get("/health")
        async def health() -> ResponseDict:
            return {"status": "ok"}

        @app.get("/ready")
        async def ready() -> ResponseDict:
            try:
                predictor._ensure_setup()
            except Exception as exc:
                return JSONResponse(
                    status_code=503, content={"status": "not ready", "error": str(exc)}
                )
            return {"status": "ready"}

        # /session — provider-credential guarded handshake.
        session_kwargs: dict[str, Any] = {}
        if self._session_auth_dep is not None:
            session_kwargs["dependencies"] = [self._session_auth_dep]

        @app.post("/session", **session_kwargs)
        async def create_session(payload: dict) -> ResponseDict:
            try:
                req = InteractiveSessionCreateRequest.from_dict(payload)
            except (KeyError, TypeError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"invalid session request: {exc}"
                ) from exc
            ttl = req.ttl_seconds or self.token_ttl_seconds
            session_id = str(req.session_id)
            token, expires_at = store.create(session_id, ttl)
            return InteractiveSessionCreateResponse(
                token=token,
                expires_at=expires_at.isoformat(),
                session_ref=session_id,
                predict_url=self.public_url,
            ).to_dict()

        # /session/{session_id}/complete — provider-credential guarded; clean up.
        # Called by the Anno server when the session is committed or discarded to
        # free the cached image promptly instead of waiting for the token TTL.
        @app.post("/session/{session_id}/complete", **session_kwargs)
        async def complete_session(session_id: str) -> ResponseDict:
            store.delete(session_id)
            return {"status": "completed", "session_id": session_id}

        # /{session_id}/infer_image — token guarded; cache + embed once.
        @app.post("/{session_id}/infer_image")
        async def infer_image(
            session_id: str,
            image: UploadFile,
            metadata: Annotated[str | None, Form()] = None,
            token: Annotated[str | None, Header(alias=token_header)] = None,
        ) -> ResponseDict:
            _authed_session(session_id, token)
            image_bytes = await image.read()
            meta_dict = None
            if metadata:
                try:
                    meta_dict = json.loads(metadata)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=400, detail=f"invalid metadata JSON: {exc}"
                    ) from exc
            try:
                state = predictor.embed(image_bytes, meta_dict)
            except Exception as exc:
                logger.exception("embed_image failed for session %s", session_id)
                raise HTTPException(status_code=500, detail=f"embed failed: {exc}") from exc
            store.set_image(session_id, state)
            return {"status": "cached", "session_id": session_id}

        # /{session_id}/predict — token guarded; prompts only, reuse cached image.
        @app.post("/{session_id}/predict")
        async def predict(
            session_id: str,
            payload: dict,
            token: Annotated[str | None, Header(alias=token_header)] = None,
        ) -> ResponseDict:
            sess = _authed_session(session_id, token)
            if not sess.has_image:
                raise HTTPException(
                    status_code=409,
                    detail="no image cached for this session; POST /infer_image first",
                )
            try:
                return predictor.serve_predict(sess.image_state, payload)
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("predict failed for session %s", session_id)
                raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc

        @app.delete("/{session_id}")
        async def end_session(
            session_id: str,
            token: Annotated[str | None, Header(alias=token_header)] = None,
        ) -> ResponseDict:
            _authed_session(session_id, token)
            store.delete(session_id)
            return {"status": "deleted", "session_id": session_id}

    # -- lifecycle ------------------------------------------------------------

    def serve_forever(self) -> None:
        logger.info(
            "anno-serve-interactive on http://%s:%d, session-auth=%s, docs=/docs",
            self.host, self.port, "on" if self._session_auth_dep else "off",
        )
        uvicorn.run(self._app, host=self.host, port=self.port, log_config=None)


def create_interactive_app(
    predictor: InteractivePredictor, **kwargs
) -> InteractiveInferenceServer:
    """Convenience alias for ``InteractiveInferenceServer(predictor, **kwargs)``."""
    return InteractiveInferenceServer(predictor, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _import_predictor(spec: str) -> InteractivePredictor:
    if ":" not in spec:
        raise ValueError(f"Expected 'module:ClassName', got {spec!r}")
    module_path, class_name = spec.split(":", 1)
    call = class_name.endswith("()")
    if call:
        class_name = class_name[:-2]
    __import__(module_path)
    module = sys.modules[module_path]
    cls = getattr(module, class_name)
    instance = cls() if isinstance(cls, type) else cls
    if not isinstance(instance, InteractivePredictor):
        raise TypeError(f"{instance!r} is not an InteractivePredictor instance")
    return instance


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="anno-sdk interactive inference server")
    p.add_argument("--predictor", required=True, help="module:ClassName (e.g. my_app:MyModel)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--auth-header", default=None, help="Provider credential header on /session")
    p.add_argument("--auth-header-value", default=None, metavar="VALUE")
    p.add_argument("--auth-query", default=None, help="Provider credential query on /session")
    p.add_argument("--auth-query-value", default=None, metavar="VALUE")
    p.add_argument("--token-ttl", type=int, default=3600, help="Fallback token TTL (seconds)")
    p.add_argument("--token-header", default="X-Session-Token")
    p.add_argument("--public-url", default=None, help="Browser-reachable base URL (predict_url)")
    p.add_argument(
        "--cors-origin",
        default=None,
        help="Allowed CORS origin for browser direct calls (e.g. 'http://localhost:5173')",
    )

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.auth_header and not args.auth_header_value:
        p.error("--auth-header-value is required when --auth-header is set")
    if args.auth_query and not args.auth_query_value:
        p.error("--auth-query-value is required when --auth-query is set")

    _require_server_extras()
    server = InteractiveInferenceServer(
        _import_predictor(args.predictor),
        host=args.host,
        port=args.port,
        auth_header=args.auth_header,
        auth_header_value=args.auth_header_value,
        auth_query=args.auth_query,
        auth_query_value=args.auth_query_value,
        token_ttl_seconds=args.token_ttl,
        token_header=args.token_header,
        public_url=args.public_url,
        cors_origin=args.cors_origin,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
