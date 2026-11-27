FROM python:3.12-slim-trixie

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      bash ca-certificates curl git jq procps poppler-utils postgresql-client \
      texlive-bibtex-extra texlive-fonts-recommended texlive-latex-base \
      texlive-latex-extra texlive-latex-recommended \
 && rm -rf /var/lib/apt/lists/* /usr/share/doc/* /usr/share/man/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --all-extras --no-dev --no-install-project \
 && uv pip install --python /opt/venv/bin/python --no-cache "numpy==2.4.2" "scipy==1.17.1" matplotlib langfuse \
 && rm -rf /root/.cache

RUN useradd -m runner
USER runner
WORKDIR /work
ENV PYTHONPATH=/pkg/src MPLBACKEND=Agg
