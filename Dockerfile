# Multi-arch image (works on amd64 + arm64). The Pi target is arm64.
FROM python:3.13-slim AS base

# ffmpeg is required at runtime (LGPL system binary, not bundled into the wheel).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible Python deps.
RUN pip install --no-cache-dir uv==0.11.5

WORKDIR /app
COPY pyproject.toml uv.lock* README.md LICENSE ./
COPY src ./src

RUN uv sync --frozen --no-dev || uv sync --no-dev

ENV MMRAG_DATA_DIR=/data \
    MMRAG_API_HOST=0.0.0.0 \
    MMRAG_API_PORT=8765
VOLUME ["/data"]
EXPOSE 8765

# Default to running both the API and the worker. Compose can override CMD
# to split them across containers if you want clean process separation.
CMD ["sh", "-c", "uv run mmrag init-db && uv run mmrag worker & uv run mmrag serve-api"]
