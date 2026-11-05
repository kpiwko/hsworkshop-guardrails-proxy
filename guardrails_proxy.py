#!/usr/bin/env python3
"""Bounded concurrent guardrails proxy.

Async (Starlette + HTTPX + Uvicorn) guardrails proxy for the workshop
OpenWebUI deployment. See README.md.

Guardrail contract preserved from the original proxy:
- ``POST /v1/chat/completions`` goes to the TrustyAI GuardrailsOrchestrator
  ``/all`` route; a blocked prompt (empty ``choices`` plus an
  ``UNSUITABLE_INPUT`` warning) is converted into a proper assistant message
  with bilingual ``BLOCKED_CONTENT``.
- Blocked requests are tagged in Langfuse via OTLP/HTTP-JSON (bounded queue,
  one exporter, fail-open: telemetry never bypasses enforcement).
- Only ``authorization``, ``content-type`` and ``accept`` headers are
  forwarded upstream.

Timeout model (see README for the full table):
- Discovery (``GET /v1/models``) deadline
  (``GUARDRAILS_PROXY_DISCOVERY_TIMEOUT_SECONDS``, 10 s).
- Request-body upload (``GUARDRAILS_PROXY_BODY_SECONDS``, 10 s, bounded
  size; slow uploads get 408, oversized 413).
- Inference — upstream headers, TTFT/first SSE event, and full
  generation, streaming or not — is governed by the total request budget
  (``GUARDRAILS_PROXY_TOTAL_SECONDS``, 300 s) plus HTTPX idle/read
  (``GUARDRAILS_PROXY_IDLE_SECONDS``, 120 s), never by a 10 s deadline.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import time
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

GUARDRAILS_URL_DEFAULT = (
    "http://guardrails-orchestrator-service.hsworkshop.svc.cluster.local:8090/all"
)
PREDICTOR_URL_DEFAULT = (
    "http://qwen36-service-predictor.hsworkshop.svc.cluster.local:8080"
)

MSG_EN = (
    "\u26a0\ufe0f Your message was blocked because it contained inappropriate language. "
    "Please rephrase your question without swear words."
)
MSG_CS = (
    "\u26a0\ufe0f V\u00e1\u0161 dotaz byl zablokov\u00e1n, proto\u017ee obsahoval nevhodn\u00fd v\u00fdraz. "
    "Zkuste ho p\u0159eformulovat bez vulgarit."
)
BLOCKED_CONTENT = f"{MSG_EN}\n\n{MSG_CS}"

PASSTHROUGH_HEADERS = {"authorization", "content-type", "accept"}

_EVENT_SEPARATORS = (b"\r\n\r\n", b"\n\n", b"\r\r")
_LINE_SPLIT = re.compile(rb"\r\n|\r|\n")


class _ResponseTooLarge(Exception):
    """Upstream body/prefix exceeded its configured bound (fail closed)."""


def _deny_all_jar():
    """Stdlib CookieJar with a deny-all policy, for HTTPX ``cookies=``.

    The HTTPX clients are long-lived and shared across users; upstream
    ``Set-Cookie`` values must never be persisted or replayed to another
    user's request. Passing a raw ``http.cookiejar.CookieJar`` (not an
    ``httpx.Cookies`` instance, which HTTPX would copy into a fresh
    default-policy jar) means the client uses this jar directly:
    ``CookieJar.extract_cookies`` consults ``set_ok`` (storage denied) and
    ``add_cookie_header`` consults ``return_ok``/``domain_return_ok``/
    ``path_return_ok`` (replay denied).
    """
    from http.cookiejar import CookieJar, DefaultCookiePolicy

    class _DenyPolicy(DefaultCookiePolicy):
        def set_ok(self, cookie, request):
            return False

        def return_ok(self, cookie, request):
            return False

        def domain_return_ok(self, domain, request):
            return False

        def path_return_ok(self, path, request):
            return False

    jar = CookieJar()
    jar.set_policy(_DenyPolicy())
    return jar


async def _run_cleanup(cleanup) -> None:
    """Run an async cleanup bounded in time without orphaning tasks.

    ``wait_for(shield(task))`` alone would leave the shielded task running
    as an orphan on timeout; here the task is always cancelled and awaited,
    and the cleanup itself releases its resources in a ``finally`` so a
    cancellation mid-``aclose`` cannot leak the chat slot.
    """
    task = asyncio.create_task(cleanup())
    try:
        await asyncio.wait_for(asyncio.shield(task), 5.0)
    except BaseException:
        task.cancel()
    try:
        await task
    except BaseException:
        pass


def _env_limit(name: str, default: float, kind=float) -> float:
    raw = os.environ.get(name)
    value: float = default
    if raw not in (None, ""):
        try:
            value = kind(raw)
        except (TypeError, ValueError):
            raise RuntimeError(f"{name} must be a positive finite number") from None
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise RuntimeError(f"{name} must be a positive finite number")
    return value


def _is_blocked(data) -> bool:
    if not isinstance(data, dict):
        return False
    choices = data.get("choices", None)
    warnings = data.get("warnings") or []
    return (
        isinstance(choices, list)
        and len(choices) == 0
        and isinstance(warnings, list)
        and any(
            isinstance(w, dict) and w.get("type") == "UNSUITABLE_INPUT"
            for w in warnings
        )
    )


def _extract_last_user_message(messages) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _forward_headers(request) -> dict:
    # Fresh dict each call; the shared HTTPX clients are never mutated.
    return {k: v for k, v in request.headers.items() if k in PASSTHROUGH_HEADERS}


def _blocked_json(model: str) -> bytes:
    return json.dumps({
        "id": "chatcmpl-blocked",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": BLOCKED_CONTENT},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }).encode()


def _blocked_sse(model: str) -> bytes:
    c1 = json.dumps({
        "id": "chatcmpl-blocked", "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": BLOCKED_CONTENT}, "finish_reason": None}],
    })
    c2 = json.dumps({
        "id": "chatcmpl-blocked", "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    return f"data: {c1}\n\ndata: {c2}\n\ndata: [DONE]\n\n".encode()


def _proxy_error(status: int, message: str, error_type: str) -> JSONResponse:
    # Static messages only: no prompt, credential, or raw exception content.
    return JSONResponse(
        {"error": {"message": message, "type": error_type}}, status_code=status
    )


def _find_event_end(buf: bytes):
    best = None
    for sep in _EVENT_SEPARATORS:
        idx = buf.find(sep)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, len(sep))
    return best


def _event_data(event: bytes):
    """Return the joined ``data:`` payload of one complete SSE event.

    Comment lines (leading ``:``) are ignored. Returns ``None`` when the event
    carries no data lines. The event is decoded only after it is complete, so
    UTF-8 sequences fragmented across network chunks parse correctly.
    """
    data_lines = []
    for line in _LINE_SPLIT.split(event):
        if line.startswith(b":"):
            continue
        if line.startswith(b"data:"):
            payload = line[len(b"data:"):]
            if payload.startswith(b" "):
                payload = payload[1:]
            data_lines.append(payload)
    if not data_lines:
        return None
    return b"\n".join(data_lines).decode("utf-8", errors="replace")


async def _read_first_event(iterator, limit: int):
    """Read upstream SSE until the first event carrying a ``data:`` line.

    Returns ``(prefix_bytes, first_data)`` where ``prefix_bytes`` is every
    byte consumed so far (relayed exactly once) and ``first_data`` is the
    decoded data payload (``None`` if the stream ended without one).
    Raises ``_ResponseTooLarge`` when no data event is found within
    ``limit`` bytes total (fail closed).
    """
    buf = bytearray()
    consumed = bytearray()
    while True:
        end = _find_event_end(buf)
        if end is not None:
            idx, seplen = end
            event = bytes(buf[: idx + seplen])
            del buf[: idx + seplen]
            consumed.extend(event)
            data = _event_data(event)
            if data is not None:
                return bytes(consumed) + bytes(buf), data
            # Comment/keep-alive event consumed. The prefix bound applies to
            # total bytes read before the first data event, so oversized
            # comment runs (even well-formed ones) fail closed.
            if len(consumed) + len(buf) > limit:
                raise _ResponseTooLarge()
            continue
        if len(consumed) + len(buf) > limit:
            # No data event within the prefix bound (unterminated garbage).
            raise _ResponseTooLarge()
        try:
            chunk = await iterator.__anext__()
        except StopAsyncIteration:
            data = _event_data(bytes(buf)) if buf else None
            return bytes(consumed) + bytes(buf), data
        buf.extend(chunk)


async def _bounded_read(response, limit: int) -> bytes:
    """Stream-read an upstream body with a hard byte bound (no buffering)."""
    data = bytearray()
    async for chunk in response.aiter_bytes():
        data.extend(chunk)
        if len(data) > limit:
            raise _ResponseTooLarge()
        # Yield control so slow bodies cannot starve the event loop.
        await asyncio.sleep(0)
    return bytes(data)


def _otel_attr(key: str, value) -> dict:
    if isinstance(value, list):
        return {"key": key, "value": {"arrayValue": {"values": [{"stringValue": str(v)} for v in value]}}}
    return {"key": key, "value": {"stringValue": str(value)}}


class _LangfuseExporter:
    """Fire-and-forget OTLP exporter for blocked requests (Langfuse v4).

    Bounded queue, a single background exporter, per-export timeout, and a
    bounded shutdown flush. Drops on overflow and fails open: telemetry
    problems never block or bypass guardrail enforcement.
    """

    def __init__(self, host: str, public_key: str, secret_key: str,
                 queue_size: int, export_seconds: float, flush_seconds: float):
        self._host = host.rstrip("/")
        self._public_key = public_key
        self._secret_key = secret_key
        self._export_seconds = export_seconds
        self._flush_seconds = flush_seconds
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

    def bind(self, client: httpx.AsyncClient) -> None:
        self._client = client

    def record_blocked(self, user_message: str, model: str) -> None:
        now_ns = time.time_ns()
        span = {
            "traceId": os.urandom(16).hex(),
            "spanId": os.urandom(8).hex(),
            "name": "blocked-request",
            "kind": 1,
            "startTimeUnixNano": str(now_ns),
            "endTimeUnixNano": str(now_ns + 1_000_000),
            "attributes": [
                _otel_attr("langfuse.trace.name", "blocked-request"),
                _otel_attr("langfuse.trace.tags", ["blocked"]),
                _otel_attr("langfuse.observation.type", "guardrail"),
                _otel_attr("langfuse.observation.level", "WARNING"),
                _otel_attr("langfuse.observation.input", json.dumps({"user_message": user_message}, ensure_ascii=False)),
                _otel_attr("langfuse.observation.output", "BLOCKED"),
                _otel_attr("langfuse.trace.metadata.reason", "UNSUITABLE_INPUT"),
                _otel_attr("langfuse.trace.metadata.model", model),
            ],
        }
        payload = json.dumps({"resourceSpans": [{
            "resource": {"attributes": [_otel_attr("service.name", "guardrails-proxy")]},
            "scopeSpans": [{"scope": {"name": "guardrails-proxy"}, "spans": [span]}],
        }]}).encode()
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # drop: telemetry must never delay or bypass enforcement

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            payload = await self._queue.get()
            await self._send(payload)

    async def _send(self, payload: bytes) -> None:
        creds = base64.b64encode(
            f"{self._public_key}:{self._secret_key}".encode()
        ).decode()
        try:
            await self._client.post(
                f"{self._host}/api/public/otel/v1/traces",
                content=payload,
                headers={
                    "Authorization": f"Basic {creds}",
                    "Content-Type": "application/json",
                    "x-langfuse-ingestion-version": "4",
                },
                timeout=self._export_seconds,
            )
        except Exception:
            print("[langfuse] blocked-request export failed", flush=True)

    async def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except BaseException:
                pass  # cancelled or crashed: shutdown must not raise
        deadline = time.monotonic() + self._flush_seconds
        while time.monotonic() < deadline:
            try:
                payload = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self._send(payload)


class _StreamRelayResponse(StreamingResponse):
    """StreamingResponse whose cleanup is owned by the ASGI response
    lifecycle, not solely by the body generator.

    The generator's ``finally`` still closes upstream, but if the generator
    is never started (client disconnect before first iteration) or is
    interrupted mid-``await``, the unconditional ``finally`` around
    ``__call__`` still closes the upstream response (shielded and bounded)
    and releases the chat slot exactly once. All cleanup helpers are
    idempotent.
    """

    def __init__(self, iterator_factory, cleanup, **kwargs):
        super().__init__(None, **kwargs)
        self._iterator_factory = iterator_factory
        self._cleanup = cleanup
        self._finalized = False
        self._body_started = False

    async def stream_response(self, send):
        if not self._body_started:
            self._body_started = True
            self.body_iterator = self._iterator_factory()
        await super().stream_response(send)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Unconditional, idempotent, cancellation-safe: even a client
            # disconnect before the first body iteration, or cancellation
            # mid-``aclose``, cannot leak the upstream connection or the
            # chat slot. Bounded so a hung upstream close cannot hold the
            # slot forever; the bound never orphans the cleanup task.
            await _run_cleanup(self._cleanup)


class _ChatCleanup:
    """Idempotent per-chat cleanup: close upstream, release the slot.

    The release runs in a ``finally`` so cancelling the upstream close
    still frees the chat slot.
    """

    def __init__(self, release, state):
        self._release = release
        self._state = state

    async def __call__(self):
        try:
            resp = self._state.get("upstream")
            if resp is not None and not resp.is_closed:
                try:
                    await resp.aclose()
                except Exception:
                    pass
            self._state["upstream"] = None
        finally:
            self._release()


def create_app() -> Starlette:
    """Build the guardrails proxy Starlette application.

    Configuration comes from simple environment variables, validated as
    positive finite numbers at creation time (no config framework).
    """
    cfg = {
        "guardrails_url": os.environ.get("GUARDRAILS_URL", GUARDRAILS_URL_DEFAULT).rstrip("/"),
        "predictor_url": os.environ.get("PREDICTOR_URL", PREDICTOR_URL_DEFAULT).rstrip("/"),
        "max_chats": int(_env_limit("GUARDRAILS_PROXY_MAX_CHATS", 4)),
        "max_body_bytes": int(_env_limit("GUARDRAILS_PROXY_MAX_BODY_BYTES", 1024 * 1024)),
        "discovery_timeout_seconds": _env_limit("GUARDRAILS_PROXY_DISCOVERY_TIMEOUT_SECONDS", 10.0),
        "body_seconds": _env_limit("GUARDRAILS_PROXY_BODY_SECONDS", 10.0),
        "max_json_bytes": int(_env_limit("GUARDRAILS_PROXY_MAX_JSON_BYTES", 4 * 1024 * 1024)),
        "max_prefix_bytes": int(_env_limit("GUARDRAILS_PROXY_MAX_PREFIX_BYTES", 64 * 1024)),
        "connect_seconds": _env_limit("GUARDRAILS_PROXY_CONNECT_SECONDS", 5.0),
        "idle_seconds": _env_limit("GUARDRAILS_PROXY_IDLE_SECONDS", 120.0),
        "total_seconds": _env_limit("GUARDRAILS_PROXY_TOTAL_SECONDS", 300.0),
        "langfuse_queue": int(_env_limit("GUARDRAILS_PROXY_LANGFUSE_QUEUE", 16)),
        "langfuse_export_seconds": _env_limit("GUARDRAILS_PROXY_LANGFUSE_EXPORT_SECONDS", 2.0),
        "langfuse_flush_seconds": _env_limit("GUARDRAILS_PROXY_LANGFUSE_FLUSH_SECONDS", 2.0),
    }
    langfuse_host = os.environ.get("LANGFUSE_HOST", "")
    langfuse_public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    langfuse_enabled = bool(langfuse_host and langfuse_public and langfuse_secret)
    exporter = (
        _LangfuseExporter(
            langfuse_host, langfuse_public, langfuse_secret,
            cfg["langfuse_queue"], cfg["langfuse_export_seconds"],
            cfg["langfuse_flush_seconds"],
        )
        if langfuse_enabled
        else None
    )
    chat_semaphore = asyncio.Semaphore(cfg["max_chats"])

    async def healthz(request):
        # App-local: never touches upstreams or the chat admission semaphore.
        if request.url.query:
            return _proxy_error(400, "query parameters are not supported", "invalid_request")
        return JSONResponse({"status": "ok"})

    async def models(request):
        if request.url.query:
            return _proxy_error(400, "query parameters are not supported", "invalid_request")
        client = request.app.state.discovery_client
        try:
            # Discovery deadline only applies to GET /v1/models.
            async with asyncio.timeout(cfg["discovery_timeout_seconds"]):
                resp = await client.send(
                    client.build_request(
                        "GET", f"{cfg['predictor_url']}/v1/models",
                        headers=_forward_headers(request),
                    ),
                    stream=True,
                )
                body = await _bounded_read(resp, cfg["max_json_bytes"])
        except (TimeoutError, httpx.TimeoutException):
            return _proxy_error(504, "upstream timeout", "upstream_timeout")
        except httpx.DecodingError:
            # DecodingError is a RequestError, not a TransportError: it must
            # be caught explicitly or it would surface as a raw 500.
            return _proxy_error(502, "upstream protocol error", "upstream_error")
        except httpx.TransportError:
            return _proxy_error(502, "upstream unavailable", "upstream_error")
        except _ResponseTooLarge:
            return _proxy_error(502, "upstream response too large", "upstream_error")
        finally:
            resp_close = locals().get("resp")
            if resp_close is not None and not resp_close.is_closed:
                try:
                    await resp_close.aclose()
                except Exception:
                    pass
        # Preserve the upstream status exactly (2xx/3xx/4xx/5xx), no redirect
        # following and no forcing to 200.
        return Response(
            content=body,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def chat(request):
        if request.url.query:
            return _proxy_error(400, "query parameters are not supported", "invalid_request")
        # Immediate 503 when all chat slots are held; acquired before the
        # request body is read, held through response lifecycle cleanup.
        if chat_semaphore.locked():
            return JSONResponse(
                {"error": {"message": "too many concurrent chats, retry shortly", "type": "overloaded"}},
                status_code=503,
                headers={"Retry-After": "1"},
            )
        await chat_semaphore.acquire()
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                chat_semaphore.release()

        state = {"upstream": None}
        cleanup = _ChatCleanup(release, state)

        try:
            response = await _chat(request, state, cfg, exporter, cleanup)
        except BaseException:
            # Run cleanup but PRESERVE the original exception/cancellation
            # (never swallow an outer CancelledError into a clean return).
            try:
                await _run_cleanup(cleanup)
            finally:
                raise
        if not isinstance(response, _StreamRelayResponse):
            # Plain (non-streaming) responses own no lifecycle hook: the
            # upstream body is already fully buffered, so cleanup runs now.
            await cleanup()
        return response

    async def _chat(request, state, cfg, exporter, cleanup):
        client = request.app.state.chat_client

        # --- bounded, time-limited request body ---
        # The disconnect watcher starts only after the body has been fully
        # read: request.stream() and request.receive() must not race for the
        # same ASGI messages.
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                declared_len = int(declared)
            except ValueError:
                return _proxy_error(400, "invalid content-length", "invalid_request")
            if declared_len > cfg["max_body_bytes"]:
                return _proxy_error(413, "request body too large", "request_too_large")
        body = bytearray()
        try:
            async with asyncio.timeout(cfg["body_seconds"]):
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > cfg["max_body_bytes"]:
                        return _proxy_error(413, "request body too large", "request_too_large")
        except (TimeoutError, asyncio.TimeoutError):
            # Slow-loris body: the slot must not be held forever.
            return _proxy_error(408, "request body timeout", "request_timeout")
        raw = bytes(body)

        try:
            req_json = json.loads(raw)
        except Exception:
            req_json = {}
        if not isinstance(req_json, dict):
            req_json = {}
        stream = bool(req_json.get("stream", False))
        model = req_json.get("model", "")
        if not isinstance(model, str):
            model = ""
        user_message = _extract_last_user_message(req_json.get("messages"))

        started = time.monotonic()
        # Watch for client disconnect during the pre-response phases
        # (upstream wait, first-event inspection, JSON body download).
        # The watcher only calls request.receive() after the request body
        # has been consumed above, and it is cancelled and awaited before
        # the streaming response takes ownership of ``receive`` — the two
        # never race for ASGI messages.
        watched = asyncio.create_task(
            _upstream_phase(request, client, state, cfg, raw, exporter,
                            user_message, model, stream, started, cleanup)
        )
        watcher = asyncio.create_task(_watch_disconnect(request))
        try:
            done, pending = await asyncio.wait(
                {watched, watcher}, return_when=asyncio.FIRST_COMPLETED
            )
            # Normal completion: cancel and await the watcher before the
            # response lifecycle takes ownership of receive() — no orphan,
            # no concurrent receive().
            watcher.cancel()
            try:
                await watcher
            except BaseException:
                pass
        except BaseException:
            # External cancellation of this coroutine while waiting (or
            # while reaping the watcher): cancel and await BOTH tasks — no
            # orphans — run cleanup, and preserve the original exception.
            # The normal watcher-disconnect path does NOT land here; it
            # completes asyncio.wait above and returns 499 below.
            for task in (watcher, watched):
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
            await _run_cleanup(cleanup)
            raise
        if watcher in done:
            # Client disconnected before the response started: cancel the
            # upstream phase (closing the upstream response via cleanup).
            watched.cancel()
            try:
                await watched
            except BaseException:
                pass
            await _run_cleanup(cleanup)
            return _proxy_error(499, "client disconnected", "client_closed")
        return watched.result()

    async def _watch_disconnect(request) -> None:
        """Complete (cancelling the upstream phase) when the client hangs up
        before the response has started."""
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    async def _upstream_phase(request, client, state, cfg, raw, exporter,
                              user_message, model, stream, started, cleanup):
        upstream_req = client.build_request(
            "POST",
            f"{cfg['guardrails_url']}/v1/chat/completions",
            content=raw,
            headers=_forward_headers(request),
        )
        try:
            # The 10 s deadline covers ONLY the request body upload (applied
            # in _chat above). Upstream headers, TTFT/first SSE event, and
            # full generation are bounded by the remaining TOTAL request
            # budget plus HTTPX connect/idle timeouts — never by 10 s.
            remaining_total = max(0.0, started + cfg["total_seconds"] - time.monotonic())
            if remaining_total <= 0:
                return _proxy_error(504, "total request budget exhausted", "upstream_timeout")
            async with asyncio.timeout(remaining_total):
                resp = await client.send(upstream_req, stream=True)
                state["upstream"] = resp

                # Preserve the upstream HTTP status exactly (3xx included;
                # redirects are never followed) before any headers are sent.
                if resp.status_code >= 300:
                    error_body = await _bounded_read(resp, cfg["max_json_bytes"])
                    return Response(
                        content=error_body,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"),
                    )

                content_type = resp.headers.get("content-type", "")

                if "text/event-stream" in content_type:
                    iterator = resp.aiter_bytes()
                    try:
                        prefix, first_data = await _read_first_event(
                            iterator, cfg["max_prefix_bytes"]
                        )
                    except _ResponseTooLarge:
                        return _proxy_error(
                            502, "upstream stream exceeded prefix bound", "upstream_error"
                        )
                    blocked = False
                    if first_data is not None:
                        try:
                            blocked = _is_blocked(json.loads(first_data))
                        except Exception:
                            blocked = False
                    if blocked:
                        if exporter is not None:
                            exporter.record_blocked(user_message, model)
                        return Response(
                            content=_blocked_sse(model),
                            status_code=200,
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache"},
                        )
                    # Allowed stream: the first meaningful event was already
                    # inspected (above) BEFORE headers are committed. Relay
                    # the prefix bytes exactly once, then continue with
                    # incremental backpressure. Mid-stream failure or client
                    # disconnect aborts; no fabricated DONE is appended. The
                    # overall stream is bounded by the total request budget.
                    total_deadline = started + cfg["total_seconds"]

                    async def relay():
                        try:
                            if prefix:
                                yield prefix
                            async for chunk in iterator:
                                if time.monotonic() > total_deadline:
                                    # Total budget exhausted: abort with a
                                    # controlled transport-style error (no
                                    # fabricated DONE, no raw exception logs).
                                    raise httpx.ReadTimeout(
                                        "total request budget exhausted"
                                    )
                                yield chunk
                        finally:
                            # Best-effort close here; the response-lifecycle
                            # cleanup remains the owner of last resort.
                            try:
                                await resp.aclose()
                            except Exception:
                                pass

                    return _StreamRelayResponse(
                        relay,
                        cleanup,
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache"},
                    )
                try:
                    resp_body = await _bounded_read(resp, cfg["max_json_bytes"])
                except _ResponseTooLarge:
                    return _proxy_error(
                        502, "upstream response too large", "upstream_error"
                    )
                blocked = False
                try:
                    blocked = _is_blocked(json.loads(resp_body))
                except Exception:
                    blocked = False
                if blocked:
                    if exporter is not None:
                        exporter.record_blocked(user_message, model)
                    if stream:
                        return Response(
                            content=_blocked_sse(model),
                            status_code=200,
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache"},
                        )
                    return Response(
                        content=_blocked_json(model),
                        status_code=200,
                        media_type="application/json",
                    )
                return Response(
                    content=resp_body,
                    status_code=resp.status_code,
                    media_type=content_type or "application/json",
                )
        except (TimeoutError, asyncio.TimeoutError):
            return _proxy_error(504, "guardrails timeout", "upstream_timeout")
        except httpx.TimeoutException:
            return _proxy_error(504, "guardrails timeout", "upstream_timeout")
        except httpx.DecodingError:
            return _proxy_error(502, "upstream protocol error", "upstream_error")
        except httpx.TransportError:
            # Includes RemoteProtocolError: abrupt mid-stream close.
            return _proxy_error(502, "guardrails unavailable", "upstream_error")
        except _ResponseTooLarge:
            return _proxy_error(502, "upstream response too large", "upstream_error")
        except ConnectionError:
            # Client disconnected before the response started (watcher).
            return _proxy_error(499, "client disconnected", "client_closed")

    @asynccontextmanager
    async def lifespan(app):
        timeout = httpx.Timeout(cfg["idle_seconds"], connect=cfg["connect_seconds"])
        # One deny-all cookie jar per client: upstream Set-Cookie values are
        # never stored or replayed across users sharing the pooled client.
        app.state.chat_client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            cookies=_deny_all_jar(),
            limits=httpx.Limits(
                max_connections=max(16, 4 * cfg["max_chats"]),
                max_keepalive_connections=max(8, 2 * cfg["max_chats"]),
            ),
        )
        app.state.discovery_client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            cookies=_deny_all_jar(),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
        export_client = None
        try:
            if exporter is not None:
                export_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(cfg["langfuse_export_seconds"]),
                    follow_redirects=False,
                    trust_env=False,
                    cookies=_deny_all_jar(),
                    limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                )
                exporter.bind(export_client)
                await exporter.start()
            yield
        finally:
            if exporter is not None:
                await exporter.shutdown()
            if export_client is not None:
                await export_client.aclose()
            await app.state.chat_client.aclose()
            await app.state.discovery_client.aclose()

    router = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    # /v1/models/ must not silently match /v1/models (no auto-redirect).
    router.router.redirect_slashes = False
    return router


# Module-level ASGI app for `uvicorn guardrails_proxy:app`.
app = create_app()
