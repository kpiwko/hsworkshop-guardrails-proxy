# Guardrails proxy image, qualified for native amd64 and arm64 builds.
# Base pinned by multi-arch INDEX digest (not a per-arch child digest); runtime code is
# installed from the full-hash requirements.lock. One Uvicorn worker on 8080,
# 20 s graceful shutdown.
# Non-root by default and OpenShift arbitrary-UID compatible: files are
# group-0 readable. Read-only-rootfs compatibility is design intent only;
# it was not separately runtime-tested.
FROM python:3.12-slim@sha256:e695ab189fa2d082b1c61c69a6b888a5adeca61ceeccad107b1ed2a125721a9e

# Install only the locked runtime dependencies (no build tools kept).
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock

# Application code; group 0 + world-read so any arbitrary UID can run it.
COPY --chown=0:0 guardrails_proxy.py /app/guardrails_proxy.py
RUN chmod g+r,o+r /app/guardrails_proxy.py

WORKDIR /app
EXPOSE 8080
ENV PYTHONDONTWRITEBYTECODE=1
# Unprivileged default; OpenShift may replace this with any assigned UID.
USER 10001:0

# Exactly one worker (the app bounds concurrency itself); 20 s drain on
# SIGTERM before Uvicorn force-closes in-flight connections.
CMD ["uvicorn", "guardrails_proxy:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-graceful-shutdown", "20"]
