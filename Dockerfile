# Multi-arch image (works on amd64 + arm64). The Pi target is arm64.
FROM python:3.13-slim AS base

# Runtime system deps:
# - ffmpeg: normalize/video/audio extraction
# - tesseract-ocr: OCR stage
# - libgl1/libglib2.0-0: OpenCV wheels imported by PySceneDetect on slim Debian
# - libgomp1: OpenMP runtime used by ML wheels such as ctranslate2
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

# uv for fast Python dependency installation.
RUN pip install --no-cache-dir uv==0.11.17

WORKDIR /app
COPY pyproject.toml uv.lock* README.md LICENSE ./
COPY src ./src

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv venv /app/.venv \
 && uv export --quiet --frozen --extra m3-visual --no-dev --no-emit-project --no-hashes --output-file /tmp/constraints.txt \
 && uv pip install --python /app/.venv/bin/python --torch-backend cpu --no-cache --constraints /tmp/constraints.txt -e ".[m3-visual]" \
 && rm /tmp/constraints.txt

ENV PATH="/app/.venv/bin:${PATH}"

ENV MMRAG_DATA_DIR=/data \
    MMRAG_API_HOST=0.0.0.0 \
    MMRAG_API_PORT=8765 \
    MMRAG_MCP_HOST=0.0.0.0 \
    MMRAG_MCP_PORT=8766 \
    MMRAG_MCP_PATH=/mcp \
    MMRAG_WORKER_CONCURRENCY=1
VOLUME ["/data"]
EXPOSE 8765 8766

# Default to the shared tailnet MCP transport. Compose splits init, MCP, and
# worker into separate services for the Pi deploy path.
CMD ["mmrag", "serve-mcp-http"]
