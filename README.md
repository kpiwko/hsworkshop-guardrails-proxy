# hsworkshop-guardrails-proxy

Bounded-concurrent async guardrails proxy (OpenWebUI → TrustyAI
GuardrailsOrchestrator) for the Hybrid Cloud / GenAI workshop. Replaces a
single-threaded inline proxy whose serialized requests wedged the readiness
probe. Starlette + HTTPX + Uvicorn, one worker per pod.

Image: `ghcr.io/kpiwko/hsworkshop-guardrails-proxy` (multi-arch
amd64/arm64), built by GitHub Actions on every push to `main`.

## What it preserves

- `POST /v1/chat/completions` routes through the TrustyAI guardrails
  gateway (`.../all`); a blocked prompt (empty `choices` plus an
  `UNSUITABLE_INPUT` warning) becomes a proper assistant message with the
  bilingual `BLOCKED_CONTENT` used by the workshop chat.
- Blocked requests are tagged in Langfuse via one OTLP/HTTP-JSON span
  (bounded queue, one exporter, bounded export/flush timeouts, disabled
  when `LANGFUSE_*` keys are absent). Telemetry drops or failures never
  bypass guardrail enforcement.
- Only `authorization`, `content-type` and `accept` headers are forwarded
  upstream.

## What it fixes

The old single-threaded `http.server` proxy serialized requests, so one
streaming chat wedged `GET /v1/models` — the readiness probe — causing
endpoint removal (`ECONNREFUSED` / model not found). This proxy is a
Starlette app served by Uvicorn with app-local `GET /healthz` that never
touches upstreams or chat admission, plus a per-pod bound of concurrent
chats (default 4, provisional safety default, not capacity).

## Routes

| Route | Upstream | Notes |
| --- | --- | --- |
| `GET /healthz` | none | app-local readiness |
| `GET /v1/models` | predictor | forwarded read |
| `POST /v1/chat/completions` | guardrails `/all` | bounded, streamed |

Unknown paths, methods, and query strings are rejected with 400/404/405
and never reach an upstream. No redirects, no retries, no predictor chat
fallback.

## Limits (simple environment variables, no config framework)

| Variable | Default | Covers |
| --- | --- | --- |
| `GUARDRAILS_PROXY_MAX_CHATS` | `4` | concurrent `POST /v1/chat/completions` per pod |
| `GUARDRAILS_PROXY_MAX_BODY_BYTES` | `1048576` (1 MiB) | request body size |
| `GUARDRAILS_PROXY_BODY_SECONDS` | `10` | request body upload time (slow uploads → 408) |
| `GUARDRAILS_PROXY_DISCOVERY_TIMEOUT_SECONDS` | `10` | `GET /v1/models` upstream call only |
| `GUARDRAILS_PROXY_MAX_JSON_BYTES` | `4194304` (4 MiB) | upstream JSON/error body size |
| `GUARDRAILS_PROXY_MAX_PREFIX_BYTES` | `65536` (64 KiB) | bytes before the first SSE data event |
| `GUARDRAILS_PROXY_CONNECT_SECONDS` | `5` | upstream TCP/TLS connect |
| `GUARDRAILS_PROXY_IDLE_SECONDS` | `120` | upstream read idle |
| `GUARDRAILS_PROXY_TOTAL_SECONDS` | `300` | whole chat: headers, TTFT/first event, and full generation |
| `GUARDRAILS_PROXY_LANGFUSE_QUEUE` | `16` | blocked-span export queue |
| `GUARDRAILS_PROXY_LANGFUSE_EXPORT_SECONDS` | `2` | one OTLP export |
| `GUARDRAILS_PROXY_LANGFUSE_FLUSH_SECONDS` | `2` | shutdown flush |

Timeout model: `GUARDRAILS_PROXY_BODY_SECONDS` (10 s) applies **only** to
uploading the request body. `GUARDRAILS_PROXY_DISCOVERY_TIMEOUT_SECONDS`
(10 s) applies **only** to the `GET /v1/models` discovery call. Upstream
response headers, time-to-first-token/first SSE event, and the entire
generation — streaming or not — are governed by the remaining total budget
(`GUARDRAILS_PROXY_TOTAL_SECONDS`, 300 s) plus HTTPX connect (5 s) and read
idle (120 s) timeouts; no 10 s deadline ever covers the inference phase.
Exceeding the total budget aborts the stream with a controlled
transport-style error, without fabricating a `data: [DONE]`.

Values must be positive finite numbers; the app refuses to start otherwise.
The chat slot is acquired before the request body is read, held through
response-lifecycle cleanup, and released in a cancellation-safe, idempotent
`finally`; overflow gets an immediate `503` with `Retry-After: 1`.

Cookie isolation: the shared pooled clients use a stdlib `CookieJar` with a
deny-all policy — upstream `Set-Cookie` values are never stored or
replayed across users (verified by a real-socket test). Upstream response
headers (including `Location`) are intentionally not forwarded; only the
response status and body are relayed, so no hop-by-hop or credential-bearing
header can leak.

## Error mapping

- Upstream HTTP error status (including 3xx) → same status relayed with
  upstream body; redirects are never followed; upstream response headers
  (including `Location`) are intentionally dropped per the allowed-header
  contract.
- Transport/protocol failure (including abrupt mid-stream closes and
  `DecodingError`) or oversized upstream body/prefix → `502`.
- Total-budget exhaustion → `504` (stream aborts, no fabricated DONE).
- Request body too large → `413`; slow request body → `408`.
- The first meaningful SSE `data:` event is inspected before response
  headers are committed; an allowed stream's prefix bytes are relayed
  exactly once with incremental backpressure. Mid-stream upstream failure
  aborts the stream — no fabricated `data: [DONE]`. Client disconnects
  (before or after first output) cancel upstream and release the chat slot
  via response-lifecycle-owned, idempotent, orphan-free cleanup.

Errors use static messages; no prompts, credentials, or raw exception
content are logged.

## Environment

Upstream endpoints (defaults match the cluster Service names):

- `GUARDRAILS_URL` — TrustyAI GuardrailsOrchestrator base (`/all` route).
- `PREDICTOR_URL` — Qwen predictor base.
- `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` — from
  `guardrails-langfuse-secret`; all optional, missing any disables tagging.

## Run locally

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock
uvicorn guardrails_proxy:app --host 127.0.0.1 --port 8080
```

## Tests

`tests/test_guardrails_proxy.py` runs the app under a real Uvicorn server
on a loopback port against synthetic fake upstreams (no cluster,
inference, or secrets). Run with:

```
pip install -r requirements-dev.txt
python -m pytest tests
```

## Image build

`.github/workflows/build.yaml` builds natively on `ubuntu-24.04` (amd64) and
`ubuntu-24.04-arm` (arm64), smoke-tests each image as an arbitrary UID
(`/healthz`) before pushing `ghcr.io/kpiwko/hsworkshop-guardrails-proxy:<sha>-<arch>`
with `GITHUB_TOKEN`, then assembles the multi-arch `:<sha>` and `:main`
manifests. Deploy by digest or `:<sha>`. The package must be set to public
in GitHub package settings for anonymous pulls.

The container runs as `10001:0` and is OpenShift arbitrary-UID compatible.
