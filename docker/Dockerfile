# Stage 1: Build stage for creating a venv with uv
FROM python:3.11.6-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ADD ./pyproject.toml ./pyproject.toml
ADD ./uv.lock ./uv.lock

RUN uv sync --frozen --no-install-project

ADD ./src ./src
RUN uv sync

EXPOSE 8000

CMD ["./.venv/bin/python", "./src/main.py"]
