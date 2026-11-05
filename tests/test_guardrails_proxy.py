"""Real-socket tests for the bounded concurrent guardrails proxy.

Runs the actual Uvicorn/Starlette app on a loopback port against synthetic
fake upstreams (guardrails orchestrator, predictor, Langfuse OTLP). No
cluster, inference service, or secrets are involved — synthetic data only.

Originally written failing-first against the old single-threaded proxy
behavior (serialized requests, readiness blocked by streaming chats);
extended after independent review (B1/I1–I5) with genuine socket-level
failure injection: abrupt mid-stream closes, fragmented multi-write UTF-8,
pre-output client disconnects, Set-Cookie leak checks, upstream header
allowlist capture, oversized bodies, and configurable tiny time budgets
(no multi-second real sleeps).
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import guardrails_proxy  # noqa: E402

MSG_EN = guardrails_proxy.MSG_EN
MSG_CS = guardrails_proxy.MSG_CS

CHAT_BODY = {
    "model": "qwen",
    "messages": [{"role": "user", "content": "hello"}],
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeUpstream:
    """Minimal synthetic HTTP/1.1 server on a real loopback socket.

    The handler may be a callable ``(method, path, body) ->
    (status, ctype, payload)`` or an object with a ``handle(conn, ctx)``
    method for raw-socket control (fragmented writes, abrupt closes,
    header capture).
    """

    def __init__(self):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = None
        self.requests = []  # (method, path, headers-dict, body)
        self.raw_handler = None  # callable(conn, method, path, headers, body)
        self.handler = lambda m, p, h, b: (200, "application/json", b"{}")

    def start(self):
        self.server = socket.create_server(("127.0.0.1", self.port))
        self.server.listen(64)
        threading.Thread(target=self._serve, daemon=True).start()

    def stop(self):
        if self.server:
            self.server.close()
            self.server = None

    def _serve(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            conn.settimeout(30)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                data += chunk
            head, _, rest = data.partition(b"\r\n\r\n")
            lines = head.split(b"\r\n")
            req_method, req_path = lines[0].decode().split(" ")[0:2]
            headers = {}
            for line in lines[1:]:
                if b":" in line:
                    k, v = line.split(b":", 1)
                    headers[k.strip().lower().decode()] = v.strip().decode()
            length = int(headers.get("content-length", "0"))
            while len(rest) < length:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                rest += chunk
            body = rest[:length]
            self.requests.append((req_method, req_path, headers, body))
            if self.raw_handler is not None:
                self.raw_handler(conn, req_method, req_path, headers, body)
                return
            status, ctype, payload = self.handler(req_method, req_path, headers, body)
            conn.sendall(
                f"HTTP/1.1 {status} OK\r\nContent-Type: {ctype}\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass


@pytest.fixture()
def guardrails():
    up = FakeUpstream()
    up.start()
    yield up
    up.stop()


@pytest.fixture()
def predictor():
    up = FakeUpstream()
    up.handler = lambda m, p, h, b: (200, "application/json", b'{"data": []}')
    up.start()
    yield up
    up.stop()


@pytest.fixture()
def langfuse():
    up = FakeUpstream()
    up.handler = lambda m, p, h, b: (200, "application/json", b"{}")
    up.start()
    yield up
    up.stop()


class ProxyServer:
    def __init__(self, guardrails_url, predictor_url, langfuse_url=None, extra_env=None):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = {
            "GUARDRAILS_URL": guardrails_url,
            "PREDICTOR_URL": predictor_url,
        }
        if langfuse_url:
            env.update({
                "LANGFUSE_HOST": langfuse_url,
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
            })
        if extra_env:
            env.update(extra_env)
        import importlib
        old = dict(os.environ)
        os.environ.update(env)
        try:
            importlib.reload(guardrails_proxy)
        finally:
            os.environ.clear()
            os.environ.update(old)
        config = uvicorn.Config(
            guardrails_proxy.app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            lifespan="on",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.02)
        else:
            raise RuntimeError("proxy server did not start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture()
def proxy(guardrails, predictor, langfuse):
    server = ProxyServer(guardrails.url, predictor.url, langfuse.url)
    yield server, guardrails, predictor, langfuse
    server.stop()


def _sse_event(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _allowed_sse() -> bytes:
    return (
        _sse_event({"choices": [{"delta": {"role": "assistant", "content": "ahoj"}}]})
        + b"data: [DONE]\n\n"
    )


def _blocked_json_body() -> bytes:
    return json.dumps({"choices": [], "warnings": [{"type": "UNSUITABLE_INPUT"}]}).encode()


def _blocked_sse() -> bytes:
    return _sse_event({"choices": [], "warnings": [{"type": "UNSUITABLE_INPUT"}]}) + b"data: [DONE]\n\n"


def _allowed_json() -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()


def _raw_chat_request(base_url, body: bytes) -> bytes:
    return (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: x\r\nContent-Type: application/json\r\n"
        b"Authorization: Bearer secret-token\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )


def _open_chat_stream(base_url, body=None):
    """Open a streaming chat on a raw socket so we can abandon it."""
    if body is None:
        body = json.dumps(dict(CHAT_BODY, stream=True)).encode()
    sock = socket.create_connection(("127.0.0.1", int(base_url.rsplit(":", 1)[1])), timeout=10)
    sock.sendall(_raw_chat_request(base_url, body))
    return sock


class BarrierChat:
    """Upstream handler that holds the response until released."""

    def __init__(self):
        self.arrived = threading.Event()
        self.release = threading.Event()

    def handler(self, method, path, headers, body):
        if path.startswith("/v1/chat/completions"):
            self.arrived.set()
            self.release.wait(timeout=30)
            return 200, "application/json", _allowed_json()
        return 200, "application/json", b"{}"


def _admission_recovers(server, timeout=5.0):
    """Poll until a fresh chat is admitted (200) or fail."""
    deadline = time.monotonic() + timeout
    r = None
    while time.monotonic() < deadline:
        r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=5)
        if r.status_code == 200:
            return r
        time.sleep(0.02)
    return r


# ---------------------------------------------------------------------------
# Concurrency bounds (the core regression: old proxy serialized everything)
# ---------------------------------------------------------------------------


def test_concurrent_chats_do_not_block_health_and_models(proxy):
    server, guardrails, predictor, langfuse = proxy
    barrier = BarrierChat()
    guardrails.handler = barrier.handler

    held = [_open_chat_stream(server.url) for _ in range(4)]

    # All four hold a chat slot at the upstream barrier.
    assert barrier.arrived.wait(timeout=5)
    # 5th chat is over the bound: immediate 503 with Retry-After.
    t0 = time.monotonic()
    r = httpx.post(f"{server.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=5)
    assert r.status_code == 503
    assert r.headers.get("retry-after") == "1"
    assert time.monotonic() - t0 < 1.0  # immediate, not queued

    # Health and discovery answer quickly while all four chats are held.
    # (The old serialized proxy failed exactly here: /v1/models timed out
    # and the Service endpoint was removed.)
    t0 = time.monotonic()
    h = httpx.get(f"{server.url}/healthz", timeout=1)
    m = httpx.get(f"{server.url}/v1/models", timeout=1)
    elapsed = time.monotonic() - t0
    assert h.status_code == 200
    assert m.status_code == 200
    assert elapsed < 1.0

    # Releasing the barrier lets the held chats finish and the slot free up.
    barrier.release.set()
    for sock in held:
        try:
            sock.settimeout(5)
            data = sock.recv(65536)
            assert b"HTTP/1.1 200" in data
        finally:
            sock.close()
    r = _admission_recovers(server)
    assert r.status_code == 200


def test_disconnect_before_first_output_releases_slot_and_upstream_fast(proxy):
    """B1/I4/F2: client hangs up while the upstream is still blocked; the
    watcher must detect the disconnect, close the upstream socket, and
    free the slot — BEFORE the upstream barrier is released. This test
    fails if the pre-output disconnect watcher is removed."""
    server, guardrails, predictor, langfuse = proxy
    upstream_eof = threading.Event()
    release = threading.Event()

    def raw_blocked(conn, method, path, headers, body):
        # Hold the upstream response open (blocked recv = waiting to write
        # the response) until the proxy itself closes this socket.
        try:
            conn.settimeout(10)
            while not release.is_set():
                if not conn.recv(4096):  # proxy closed the connection
                    break
        except Exception:
            pass
        upstream_eof.set()
        conn.close()

    guardrails.raw_handler = raw_blocked

    sock = _open_chat_stream(server.url)
    time.sleep(0.3)  # let the upstream connection establish and block
    sock.close()  # genuine pre-output disconnect on a real socket

    # The upstream socket must be closed by the proxy (not by the test):
    # the release barrier is still set, so EOF can only come from the
    # watcher-triggered cancellation+cleanup.
    assert upstream_eof.wait(timeout=2), (
        "upstream socket not closed within 2 s of pre-output disconnect"
    )
    # The slot must already be free (upstream never completed normally).
    guardrails.raw_handler = None
    guardrails.handler = lambda m, p, h, b: (200, "application/json", _allowed_json())
    r = _admission_recovers(server, timeout=5.0)
    assert r.status_code == 200
    release.set()

    # N1 regression: the NORMAL watcher-disconnect path must return a 499
    # response, not raise CancelledError (the old finally-block bug let
    # the external-cancellation branch swallow normal disconnects).
    # Driven directly against the ASGI app so the response is observable.
    async def drive_499():
        body_bytes = json.dumps(dict(CHAT_BODY, stream=True)).encode()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode()),
            ],
            "client": ("127.0.0.1", 12346),
            "server": ("127.0.0.1", server.port),
        }
        sent_request = []
        sent = []

        async def receive():
            if not sent_request:
                sent_request.append(1)
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return {"type": "http.disconnect"}  # normal disconnect

        async def send(message):
            sent.append(message)

        await guardrails_proxy.app(scope, receive, send)
        return sent

    sent = asyncio.run(drive_499())
    statuses = [m for m in sent if m["type"] == "http.response.start"]
    assert statuses, f"no response started; messages: {sent}"
    assert statuses[0]["status"] == 499, (
        f"normal disconnect must return 499, got {statuses[0]['status']}"
    )


def test_external_task_cancellation_during_preoutput_reaps_tasks(proxy):
    """F5/N1: a direct external cancellation of the chat ASGI task while it
    waits on the upstream phase must cancel and await BOTH spawned tasks
    (no orphans), close the upstream, and release the chat slot. Driven by
    cancelling the actual asyncio task, not by Uvicorn shutdown flags."""
    import asyncio as _asyncio

    server, guardrails, predictor, langfuse = proxy
    upstream_eof = threading.Event()
    chat_done = threading.Event()

    def raw_blocked(conn, method, path, headers, body):
        try:
            conn.settimeout(15)
            while True:
                if not conn.recv(4096):
                    break
        except Exception:
            pass
        upstream_eof.set()
        conn.close()

    guardrails.raw_handler = raw_blocked

    async def drive():
        # Drive the ASGI app directly so we own the handler task and can
        # cancel it externally — exactly the situation N1 guards against.
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_body_bytes())).encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", server.port),
        }
        received = []

        async def receive():
            if not received:
                received.append(1)
                return {"type": "http.request", "body": _body_bytes(), "more_body": False}
            await _asyncio.sleep(3600)  # park: no further messages
            return {"type": "http.disconnect"}

        async def send(message):
            chat_done.set()

        # Start the app task and let it park in the pre-output upstream wait.
        app_task = _asyncio.create_task(
            guardrails_proxy.app(scope, receive, send)
        )
        await _asyncio.sleep(0.5)
        assert not app_task.done(), "chat should be parked pre-output"
        # External cancellation.
        app_task.cancel()
        try:
            await app_task
        except _asyncio.CancelledError:
            pass  # preserved original cancellation, as required (F6)
        else:
            raise AssertionError("external CancelledError was swallowed")

    def _body_bytes():
        return json.dumps(dict(CHAT_BODY, stream=True)).encode()

    asyncio.run(drive())

    # The upstream connection was closed by the cancellation cleanup.
    assert upstream_eof.wait(timeout=10), (
        "upstream socket not closed after external task cancellation"
    )
    # No orphaned tasks remain from the cancelled chat.
    guardrails.raw_handler = None
    guardrails.handler = lambda m, p, h, b: (200, "application/json", _allowed_json())
    r = _admission_recovers(server)
    assert r.status_code == 200


def test_disconnect_midstream_closes_upstream_and_frees_slot(proxy):
    """B1/I4: disconnect after headers/streaming started; upstream socket
    closes (server sees EOF) and the slot is released."""
    server, guardrails, predictor, langfuse = proxy
    upstream_eof = threading.Event()

    def raw_sse(conn, method, path, headers, body):
        # Stream one event, wait for the client (proxy) to vanish, then
        # report EOF on the upstream socket.
        payload = _allowed_sse()
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            + f"Content-Length: {len(payload) + 64}\r\n".encode()  # more than we write
            + b"Connection: close\r\n\r\n"
        )
        conn.sendall(payload)
        try:
            conn.settimeout(10)
            while True:
                data = conn.recv(4096)  # blocks until proxy closes/aborts
                if not data:
                    break
        except Exception:
            pass
        upstream_eof.set()
        conn.close()

    guardrails.raw_handler = raw_sse

    sock = _open_chat_stream(server.url)
    # Read the response head (headers committed + prefix relayed = the
    # stream genuinely started), then disconnect mid-stream. All available
    # bytes may arrive with the head; do not block for more.
    sock.settimeout(5)
    head = b""
    while b"\r\n\r\n" not in head:
        head += sock.recv(65536)
    assert b"200" in head.split(b"\r\n", 1)[0]
    sock.setblocking(False)
    try:
        head += sock.recv(65536)  # any already-queued stream bytes
    except (BlockingIOError, TimeoutError):
        pass
    sock.close()

    assert upstream_eof.wait(timeout=10), "upstream socket was not closed on disconnect"
    guardrails.raw_handler = None
    guardrails.handler = lambda m, p, h, b: (200, "application/json", _allowed_json())
    r = _admission_recovers(server)
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Guardrail semantics preserved
# ---------------------------------------------------------------------------


def test_allowed_and_blocked_json(proxy):
    server, guardrails, predictor, langfuse = proxy
    guardrails.handler = lambda m, p, h, b: (200, "application/json", _allowed_json())
    r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=5)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok"

    guardrails.handler = lambda m, p, h, b: (200, "application/json", _blocked_json_body())
    r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=5)
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "inappropriate language" in content
    assert MSG_CS in content  # bilingual block message preserved
    assert r.json()["choices"][0]["finish_reason"] == "stop"


def test_blocked_upstream_sse_becomes_blocked_sse_and_tags_langfuse(proxy):
    server, guardrails, predictor, langfuse = proxy
    guardrails.handler = lambda m, p, h, b: (200, "text/event-stream", _blocked_sse())
    r = httpx.post(
        f"{server.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=5
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[len("data: "):])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert "inappropriate language" in content
    assert MSG_CS in content  # bilingual block message preserved
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert r.text.rstrip().endswith("data: [DONE]")
    # Langfuse OTLP export happened with the expected shape.
    deadline = time.monotonic() + 5
    exported = False
    while time.monotonic() < deadline:
        for method, path, hdrs, body2 in langfuse.requests:
            if path == "/api/public/otel/v1/traces":
                exported = True
                assert b"blocked" in body2
                assert b"UNSUITABLE_INPUT" in body2
                assert b"chat.hsworkshop" not in body2
        if exported:
            break
        time.sleep(0.05)
    assert exported


def test_blocked_json_upstream_with_stream_true_returns_sse(proxy):
    server, guardrails, predictor, langfuse = proxy
    guardrails.handler = lambda m, p, h, b: (200, "application/json", _blocked_json_body())
    r = httpx.post(
        f"{server.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=5
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in r.text


def test_allowed_stream_relayed_with_headers_after_first_event(proxy):
    server, guardrails, predictor, langfuse = proxy
    guardrails.handler = lambda m, p, h, b: (200, "text/event-stream", _allowed_sse())
    r = httpx.post(
        f"{server.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=5
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "ahoj" in r.text
    assert r.text.count("data: [DONE]") == 1  # relayed exactly once


# ---------------------------------------------------------------------------
# I1: pooled client cookie isolation
# ---------------------------------------------------------------------------


def test_upstream_set_cookie_is_never_stored_or_replayed(proxy):
    server, guardrails, predictor, langfuse = proxy
    # Upstream sets a cookie on the first response.
    def sets_cookie(method, path, headers, body):
        return 200, "application/json", _allowed_json()

    def raw_sets_cookie(conn, method, path, headers, body):
        payload = _allowed_json()
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Set-Cookie: session=leaky-value; Path=/\r\n"
            + f"Content-Length: {len(payload)}\r\n".encode()
            + b"Connection: close\r\n\r\n" + payload
        )
        conn.close()

    guardrails.raw_handler = raw_sets_cookie
    r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=5)
    assert r.status_code == 200

    # Second request through the same pooled client: no Cookie header may
    # appear upstream (cross-user leak).
    guardrails.raw_handler = None
    guardrails.handler = sets_cookie
    r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=5)
    assert r.status_code == 200
    method, path, hdrs, body = guardrails.requests[-1]
    assert "cookie" not in hdrs, f"Cookie replayed upstream: {hdrs.get('cookie')!r}"


def test_forwarded_headers_are_allowlisted_only(proxy):
    server, guardrails, predictor, langfuse = proxy
    guardrails.handler = lambda m, p, h, b: (200, "application/json", _allowed_json())
    r = httpx.post(
        f"{server.url}/v1/chat/completions",
        json=CHAT_BODY,
        headers={
            "Authorization": "Bearer tok",
            "X-Custom": "leak",
            "Cookie": "session=leak",
            "Accept": "application/json",
        },
        timeout=5,
    )
    assert r.status_code == 200
    method, path, hdrs, body = guardrails.requests[-1]
    # I5: assert on the actual captured upstream header dict.
    assert hdrs.get("authorization") == "Bearer tok"
    assert "x-custom" not in hdrs
    assert "cookie" not in hdrs
    assert "user-agent" not in hdrs or hdrs["user-agent"].startswith("python-httpx")


# ---------------------------------------------------------------------------
# Routing strictness
# ---------------------------------------------------------------------------


def test_unknown_path_method_and_query_rejected_without_upstream(proxy):
    server, guardrails, predictor, langfuse = proxy
    before = list(guardrails.requests)
    r = httpx.get(f"{server.url}/v1/unknown", timeout=5)
    assert r.status_code == 404
    r = httpx.get(f"{server.url}/v1/chat/completions", timeout=5)  # wrong method
    assert r.status_code == 405
    r = httpx.get(f"{server.url}/healthz?bypass=1", timeout=5)
    assert r.status_code == 400
    r = httpx.post(f"{server.url}/v1/chat/completions?x=1", json=CHAT_BODY, timeout=5)
    assert r.status_code == 400
    r = httpx.put(f"{server.url}/v1/models", timeout=5)
    assert r.status_code == 405
    r = httpx.get(f"{server.url}/v1/models/", timeout=5)  # no silent redirect
    assert r.status_code == 404
    assert list(guardrails.requests) == before  # no upstream traffic at all
    assert all(p[1] == "/v1/models" for p in predictor.requests)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_upstream_http_error_status_preserved(proxy):
    server, guardrails, predictor, langfuse = proxy
    guardrails.handler = lambda m, p, h, b: (500, "application/json", b'{"error": "boom"}')
    r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=5)
    assert r.status_code == 500
    assert b"boom" in r.content


def test_upstream_3xx_status_preserved_not_followed(proxy):
    server, guardrails, predictor, langfuse = proxy

    def raw_302(conn, method, path, headers, body):
        conn.sendall(
            b"HTTP/1.1 302 Found\r\nContent-Type: text/plain\r\n"
            b"Location: http://127.0.0.1:1/evil\r\nContent-Length: 0\r\n"
            b"Connection: close\r\n\r\n"
        )
        conn.close()

    guardrails.raw_handler = raw_302
    r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=5)
    # Status preserved exactly; the redirect is never followed. Location
    # and other upstream headers are intentionally NOT forwarded (the
    # allowed-header contract never relays upstream response headers, so
    # no hop-by-hop or credential-bearing header can leak).
    assert r.status_code == 302
    assert "location" not in r.headers
    # Exactly one upstream request: the redirect was never followed.
    assert len(guardrails.requests) == 1


def test_upstream_refused_maps_to_502(proxy):
    server, guardrails, predictor, langfuse = proxy
    guardrails.stop()
    r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=10)
    assert r.status_code == 502


def test_oversized_prefix_fails_closed_502(proxy):
    server, guardrails, predictor, langfuse = proxy

    def huge_prefix(method, path, headers, body):
        payload = b": " + b"k" * (70 * 1024) + b"\n\n" + _allowed_sse()
        return 200, "text/event-stream", payload

    guardrails.handler = huge_prefix
    r = httpx.post(
        f"{server.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=10
    )
    assert r.status_code == 502  # fail closed: no bypass, no partial stream


def test_midstream_abrupt_close_no_false_done(proxy):
    """I5: real protocol failure — server declares Content-Length larger
    than it ever writes, then closes the socket abruptly. The proxy must
    surface a protocol error and must not fabricate data: [DONE]."""
    server, guardrails, predictor, langfuse = proxy

    def raw_truncated(conn, method, path, headers, body):
        payload = _allowed_sse()
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            + f"Content-Length: {len(payload) + 4096}\r\n".encode()
            + b"Connection: close\r\n\r\n"
        )
        # Write only the first complete event, then abruptly close while
        # the proxy still expects Content-Length bytes.
        conn.sendall(_sse_event({"choices": [{"delta": {"role": "assistant", "content": "ah"}}]}))
        time.sleep(0.2)
        conn.close()  # abrupt: mid-body, mid-stream

    guardrails.raw_handler = raw_truncated
    received = b""
    with httpx.stream(
        "POST", f"{server.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=10
    ) as r:
        assert r.status_code == 200
        try:
            for chunk in r.iter_bytes():
                received += chunk
        except Exception:
            pass  # remote read error is the expected outcome
    assert b"[DONE]" not in received  # no fabricated stop signal
    assert b"ah" in received  # the genuinely relayed prefix arrived


def test_fragmented_utf8_multivrite_blocked_detection(proxy):
    """I5/F3: SSE event written in many small TCP writes, splitting the
    UTF-8 multibyte payload and line separators; blocked must still be
    detected (the Czech text in MSG_CS only round-trips if multibyte
    sequences fragmented across writes are reassembled correctly)."""
    server, guardrails, predictor, langfuse = proxy

    def raw_fragmented(conn, method, path, headers, body):
        payload = _blocked_sse()
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            + f"Content-Length: {len(payload)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
        )
        # One byte per write: maximum fragmentation of UTF-8 and CRLF.
        for i in range(len(payload)):
            conn.sendall(payload[i : i + 1])
            if i % 16 == 0:
                time.sleep(0.001)
        conn.close()

    guardrails.raw_handler = raw_fragmented
    r = httpx.post(
        f"{server.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=10
    )
    assert r.status_code == 200
    events = [
        json.loads(line[len("data: "):])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert "inappropriate language" in content  # blocked, not raw relay
    assert MSG_CS in content  # Czech multibyte survived re-encoding


def test_fragmented_utf8_allowed_payload_bytes_preserved(proxy):
    """F3: an ALLOWED stream whose SSE data event contains non-ASCII
    multibyte UTF-8 (written byte-per-write) must be relayed byte-exact —
    not only blocked-detection on ASCII payloads."""
    server, guardrails, predictor, langfuse = proxy
    czech = "P\u0159\u00edli\u0161 \u017elu\u0165ou\u010dk\u00fd k\u016f\u0148"  # přiliš žluťoučký kůň
    payload = _sse_event({"choices": [{"delta": {"role": "assistant", "content": czech}}]}) + b"data: [DONE]\n\n"
    # Sanity: the payload genuinely contains multibyte UTF-8 sequences.
    assert len(czech.encode("utf-8")) > len(czech)

    def raw_fragmented(conn, method, path, headers, body):
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            + f"Content-Length: {len(payload)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
        )
        for i in range(len(payload)):
            conn.sendall(payload[i : i + 1])
            if i % 16 == 0:
                time.sleep(0.001)
        conn.close()

    guardrails.raw_handler = raw_fragmented
    r = httpx.post(
        f"{server.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=10
    )
    assert r.status_code == 200
    assert r.content == payload  # byte-exact relay of the allowed stream


def test_comment_only_events_then_allowed(proxy):
    server, guardrails, predictor, langfuse = proxy
    payload = b": keepalive\r\n\r\n: another\n\n" + _allowed_sse()
    guardrails.handler = lambda m, p, h, b: (200, "text/event-stream", payload)
    r = httpx.post(
        f"{server.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=5
    )
    assert r.status_code == 200
    assert "ahoj" in r.text


# ---------------------------------------------------------------------------
# I2/I3: time and size bounds with tiny configurable budgets
# ---------------------------------------------------------------------------


def test_long_wait_bounded_by_total_budget_504(proxy):
    """I2: an upstream that never sends response headers is bounded by the
    TOTAL budget (not the 10 s body deadline) — with a tiny total, no real
    multi-second sleeps."""
    server, guardrails, predictor, langfuse = proxy
    tiny = ProxyServer(
        guardrails.url, predictor.url, langfuse.url,
        extra_env={"GUARDRAILS_PROXY_TOTAL_SECONDS": "0.3"},
    )
    try:
        never = threading.Event()

        def raw_hang(conn, method, path, headers, body):
            never.wait(timeout=10)  # never send headers
            conn.close()

        guardrails.raw_handler = raw_hang
        t0 = time.monotonic()
        r = httpx.post(
            f"{tiny.url}/v1/chat/completions", json=CHAT_BODY, timeout=10
        )
        elapsed = time.monotonic() - t0
        assert r.status_code == 504
        assert elapsed < 3.0  # bounded by the tiny total budget
        never.set()
        # Slot released after the budget: admission recovers.
        guardrails.raw_handler = None
        guardrails.handler = lambda m, p, h, b: (200, "application/json", _allowed_json())
        r = _admission_recovers(tiny)
        assert r.status_code == 200
    finally:
        tiny.stop()


def test_long_generation_within_total_budget_is_not_killed_by_body_deadline(proxy):
    """F1/I2: with a REAL tiny body-upload budget (BODY_SECONDS=0.3), a chat
    whose body uploads quickly but whose upstream generation is slow
    (> 0.3 s, within the total budget) must complete — for both streaming
    and non-streaming upstream responses. The body deadline must never
    cover the inference phase."""
    server, guardrails, predictor, langfuse = proxy
    tiny = ProxyServer(
        guardrails.url, predictor.url, langfuse.url,
        extra_env={"GUARDRAILS_PROXY_BODY_SECONDS": "0.3"},
    )
    try:
        # Body uploads instantly; headers immediately; generation after
        # 1.0 s — well past the 0.3 s body budget but inside the total.
        def raw_slow_stream(conn, method, path, headers, body):
            payload = _allowed_sse()
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                + f"Content-Length: {len(payload)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
            )
            time.sleep(1.0)  # generation time > body budget (0.3 s)
            conn.sendall(payload)
            conn.close()

        guardrails.raw_handler = raw_slow_stream
        r = httpx.post(
            f"{tiny.url}/v1/chat/completions", json=dict(CHAT_BODY, stream=True), timeout=10
        )
        assert r.status_code == 200
        assert "ahoj" in r.text  # streaming completed despite body budget

        # Non-streaming upstream response, equally slow.
        def raw_slow_json(conn, method, path, headers, body):
            payload = _allowed_json()
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
            )
            time.sleep(1.0)
            conn.sendall(payload)
            conn.close()

        guardrails.raw_handler = raw_slow_json
        r = httpx.post(f"{tiny.url}/v1/chat/completions", json=CHAT_BODY, timeout=10)
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "ok"
    finally:
        tiny.stop()


def test_trickling_stream_aborts_at_total_budget_without_false_done(proxy):
    """F4: with a tiny total budget, a stream that trickles forever must
    abort via the controlled total-budget error — no fabricated [DONE],
    and the slot must be released afterwards."""
    server, guardrails, predictor, langfuse = proxy
    tiny = ProxyServer(
        guardrails.url, predictor.url, langfuse.url,
        extra_env={"GUARDRAILS_PROXY_TOTAL_SECONDS": "0.5"},
    )
    try:
        def raw_trickle(conn, method, path, headers, body):
            first = _sse_event({"choices": [{"delta": {"role": "assistant", "content": "ah"}}]})
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )
            # Proper HTTP chunked framing for each write.
            def chunk(data: bytes) -> None:
                conn.sendall(hex(len(data))[2:].encode() + b"\r\n" + data + b"\r\n")

            chunk(first)
            # Trickle forever (well past the 0.5 s total budget).
            for _ in range(200):
                chunk(b": keepalive\n\n")
                time.sleep(0.05)
            conn.close()

        guardrails.raw_handler = raw_trickle
        received = b""
        with httpx.stream(
            "POST", f"{tiny.url}/v1/chat/completions",
            json=dict(CHAT_BODY, stream=True), timeout=10,
        ) as r:
            assert r.status_code == 200
            try:
                for chunk in r.iter_bytes():
                    received += chunk
            except Exception:
                pass  # aborted stream: read error is expected
        assert b"[DONE]" not in received  # no fabricated stop signal
        assert b"ah" in received  # prefix was relayed
        # Slot released after the abort: admission recovers.
        guardrails.raw_handler = None
        guardrails.handler = lambda m, p, h, b: (200, "application/json", _allowed_json())
        r = _admission_recovers(tiny)
        assert r.status_code == 200
    finally:
        tiny.stop()


def test_slow_request_body_times_out_408(proxy):
    """I3: a client that trickles its body must not hold the slot forever."""
    server, guardrails, predictor, langfuse = proxy
    tiny = ProxyServer(
        guardrails.url, predictor.url, langfuse.url,
        extra_env={"GUARDRAILS_PROXY_BODY_SECONDS": "0.3"},
    )
    try:
        sock = socket.create_connection(("127.0.0.1", tiny.port), timeout=10)
        body = json.dumps(CHAT_BODY).encode()
        head = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: x\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
        )
        sock.sendall(head)
        sock.sendall(body[:8])
        sock.settimeout(10)
        # The proxy should respond 408 without the rest of the body.
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
        assert b"408" in data.split(b"\r\n", 1)[0]
        sock.close()
        # Slot released: admission recovers.
        guardrails.handler = lambda m, p, h, b: (200, "application/json", _allowed_json())
        r = _admission_recovers(tiny)
        assert r.status_code == 200
    finally:
        tiny.stop()


def test_oversized_request_body_413(proxy):
    server, guardrails, predictor, langfuse = proxy
    big = dict(CHAT_BODY, messages=[{"role": "user", "content": "x" * (2 * 1024 * 1024)}])
    r = httpx.post(f"{server.url}/v1/chat/completions", json=big, timeout=10)
    assert r.status_code == 413


def test_oversized_json_response_502(proxy):
    server, guardrails, predictor, langfuse = proxy

    def huge_json(method, path, headers, body):
        return 200, "application/json", b'{"choices": [{"message": {"content": "' + b"y" * (5 * 1024 * 1024) + b'"}}]}'

    guardrails.handler = huge_json
    r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=10)
    assert r.status_code == 502


# ---------------------------------------------------------------------------
# Telemetry fail-open
# ---------------------------------------------------------------------------


def test_exporter_outage_does_not_change_enforcement(proxy):
    server, guardrails, predictor, langfuse = proxy
    # Langfuse endpoint refuses connections: exporter must fail open.
    langfuse.stop()
    guardrails.handler = lambda m, p, h, b: (200, "application/json", _blocked_json_body())
    r = httpx.post(f"{server.url}/v1/chat/completions", json=CHAT_BODY, timeout=5)
    assert r.status_code == 200
    assert "inappropriate language" in r.text  # enforcement unchanged
