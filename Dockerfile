# Sentinel Log AI - Multi-stage Docker Build
#
# This Dockerfile creates a production-ready container with both
# the Go agent and Python ML engine.
#
# Build: docker build -t sentinel-log-ai .
# Run:   docker run -it --rm sentinel-log-ai analyze logs/sample.jsonl
#
# Stages:
# 1. go-builder: Compiles Go agent binary
# 2. python-builder: Installs Python dependencies
# 3. runtime: Minimal production image

# ==============================================================================
# Stage 1: Go Builder
# ==============================================================================
FROM golang:1.22-alpine AS go-builder

WORKDIR /build

# Install build dependencies
RUN apk add --no-cache git make

# Copy Go module files first for layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy Go source code
COPY cmd/ cmd/
COPY internal/ internal/
COPY proto/ proto/

# Build the Go agent
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-w -s -X main.version=$(cat VERSION 2>/dev/null || echo 'dev')" \
    -o sentinel-log-ai \
    ./cmd/agent

# ==============================================================================
# Stage 2: Python Builder
# ==============================================================================
FROM python:3.12-slim AS python-builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Python package files
COPY pyproject.toml README.md ./
COPY python/ python/

# Install Python package with ML dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[ml,llm]"

# ==============================================================================
# Stage 3: Runtime
# ==============================================================================
FROM python:3.12-slim AS runtime

# Labels for container metadata
LABEL org.opencontainers.image.title="Sentinel Log AI"
LABEL org.opencontainers.image.description="AI-powered log intelligence engine"
LABEL org.opencontainers.image.vendor="Sentinel Log AI Team"
LABEL org.opencontainers.image.source="https://github.com/varadharajaan/sentinel-log-ai"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN groupadd --gid 1000 sentinel && \
    useradd --uid 1000 --gid sentinel --shell /bin/bash --create-home sentinel

WORKDIR /app

# Copy Go binary from builder
COPY --from=go-builder /build/sentinel-log-ai /usr/local/bin/

# Copy Python virtual environment from builder
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Python source (needed for package)
COPY --from=python-builder /build/python /app/python
COPY --from=python-builder /build/pyproject.toml /app/

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create directories for data
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R sentinel:sentinel /app

# Switch to non-root user
USER sentinel

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SENTINEL_LOG_LEVEL=info
ENV SENTINEL_DATA_DIR=/app/data
ENV SENTINEL_MODEL_DIR=/app/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from sentinel_ml import config; print('healthy')" || exit 1

# Default command
ENTRYPOINT ["sentinel-log-ai"]
CMD ["--help"]

# Expose gRPC port
EXPOSE 50051
