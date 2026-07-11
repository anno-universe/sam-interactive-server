FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates openssl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    NUMBA_CACHE_DIR=/tmp \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock .
COPY server.py .

RUN uv sync --no-dev --frozen

EXPOSE 8422

ENTRYPOINT ["uv", "run", "python", "server.py"]
