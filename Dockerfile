# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Build argument for development dependencies
ARG INSTALL_DEV=false

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    if [ "$INSTALL_DEV" = "true" ]; then pip install -r requirements-dev.txt; fi

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY README.md LICENSE MANIFEST.in ./

# Install the package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash novaeval && \
    chown -R novaeval:novaeval /app

# Switch to non-root user
USER novaeval

# Create directories for data and results
RUN mkdir -p /app/data /app/results

# Set default command
CMD ["novaeval", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import novaeval; print('NovaEval is healthy')" || exit 1

# Labels
LABEL maintainer="Noveum Team <team@noveum.ai>" \
      version="0.1.0" \
      description="NovaEval - AI Model Evaluation Framework" \
      org.opencontainers.image.source="https://github.com/Noveum/NovaEval" \
      org.opencontainers.image.documentation="https://noveum.github.io/NovaEval" \
      org.opencontainers.image.licenses="Apache-2.0"
