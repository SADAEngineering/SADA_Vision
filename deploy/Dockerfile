# SADA Vision - Dienst-Image
#
# Zwei Stufen, damit der Bauwerkzeugkasten nicht mitfaehrt. Im Ergebnis liegt
# kein torch: trainiert wird ausserhalb (siehe Dockerfile.train), serviert
# wird ONNX. Das sind rund 2 GB Unterschied je laufendem Container - und auf
# einem On-Prem-Blech zaehlt das.

# ---------------------------------------------------------------- Stufe 1 --
FROM python:3.12-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install .

# ---------------------------------------------------------------- Stufe 2 --
FROM python:3.12-slim-bookworm AS runtime

# opencv-headless braucht trotz "headless" die glib; libgomp zieht scikit-image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgomp1 \
      curl \
 && rm -rf /var/lib/apt/lists/*

# Kein root. Die Modelldateien gehoeren ihm, sonst scheitert ein Mount.
RUN useradd --create-home --uid 10001 vision

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SADAVISION_MODEL_DIR=/models \
    SADAVISION_PORT=8080 \
    # Ohne Deckel nimmt sich jede Bibliothek alle Kerne und sie behindern
    # sich gegenseitig. Der Wert wird in compose je nach Blech gesetzt.
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

WORKDIR /app
RUN mkdir -p /models && chown -R vision:vision /models

USER vision
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health/live || exit 1

CMD ["python", "-m", "sada_vision"]

# ------------------------------------------------------- Stufe 3: Tests ----
# Nur fuer die Pipeline und den Laptop - das Dienst-Image bleibt Stufe 2.
FROM runtime AS dev

USER root
COPY --from=build /build/pyproject.toml /app/pyproject.toml
RUN /opt/venv/bin/pip install --no-cache-dir \
      "pytest>=8.3" "pytest-asyncio>=0.24" "httpx>=0.27" "ruff>=0.7"
ENV PYTHONPATH=/app/src:/app
WORKDIR /app
CMD ["pytest", "-q"]
