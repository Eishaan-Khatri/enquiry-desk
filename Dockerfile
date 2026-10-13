# The desk as one container. Build and run it with compose:
#
#   docker compose --profile app up -d --build
#
# Dependencies are installed in a layer of their own, before the source is copied in. A change
# to the code then rebuilds in seconds, because the slow layer is reused from the cache.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# "embed" is the real embedding model. "postgres" is the PostgreSQL driver. Add "--extra voice"
# for speech, which makes the image a few hundred megabytes larger.
ARG EXTRAS="--extra embed --extra postgres"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project $EXTRAS

COPY src ./src
COPY data ./data
COPY web ./web
COPY prices.toml ./
RUN uv sync --frozen --no-dev $EXTRAS

# Run as an ordinary user. If the process is ever taken over, it is not root in the container.
RUN useradd --create-home desk && mkdir -p /app/var && chown -R desk /app/var
USER desk
ENV DESK_DATA_DIR=/app/var

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["/app/.venv/bin/uvicorn", "enquiry_desk.api:app", "--host", "0.0.0.0", "--port", "8000"]
