# ---------------------------------------------------------
# Stage 1: Builder - build wheels & heavy deps
# ---------------------------------------------------------
FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch (lightweight)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Build wheels for the rest of your dependencies
RUN pip wheel --wheel-dir /wheels -r requirements.txt


# ---------------------------------------------------------
# Stage 2: Final Runtime Image
# ---------------------------------------------------------
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime-linked libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels
COPY requirements.txt .

# Install PyTorch again (runtime)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install wheels (your dependencies)
RUN pip install --no-index --find-links=/wheels -r requirements.txt

# Copy ONLY code → into /app/src (important!)
COPY . /app/src

# Create non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Python should import from /app/src
ENV PYTHONPATH=/app/src

EXPOSE 8007

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8007/ || exit 1

# Run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8007"]
