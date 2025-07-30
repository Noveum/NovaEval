"""
NovaEval API - FastAPI application entry point.

This module provides the main FastAPI application with health endpoints
and API routing configuration.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Optional prometheus dependencies
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = generate_latest = CONTENT_TYPE_LATEST = None

from app.api.v1 import api_router
from app.core.config import get_settings
from app.core.exceptions import NovaEvalAPIError
from app.core.logging import get_logger, log_error, log_request, setup_logging


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Application lifespan manager for startup and shutdown tasks.
    """
    # Startup
    settings = get_settings()
    setup_logging(level=settings.log_level, structured=True, enable_file_logging=False)

    logger = get_logger("startup")
    logger.info("NovaEval API starting up", extra={"version": "1.0.0"})

    yield

    # Shutdown
    logger.info("NovaEval API shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title="NovaEval API",
        description="HTTP API for NovaEval evaluation framework",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Setup Prometheus metrics (optional)
    REQUEST_COUNT = None
    REQUEST_DURATION = None

    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT = Counter(
            "novaeval_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )

        REQUEST_DURATION = Histogram(
            "novaeval_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
        )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging and metrics middleware
    @app.middleware("http")
    async def log_requests_and_metrics(request: Request, call_next):
        """Log all HTTP requests, responses, and collect metrics."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_seconds = time.time() - start_time
        duration_ms = duration_seconds * 1000

        # Update metrics (skip metrics endpoint to avoid recursion)
        if (
            request.url.path != "/metrics"
            and PROMETHEUS_AVAILABLE
            and REQUEST_COUNT
            and REQUEST_DURATION
        ):
            endpoint = request.url.path
            # Simplify endpoint for metrics (remove IDs)
            if "/api/v1/" in endpoint:
                parts = endpoint.split("/")
                if len(parts) > 4 and parts[4] not in [
                    "models",
                    "datasets",
                    "scorers",
                    "evaluations",
                    "components",
                ]:
                    # Replace ID with placeholder
                    parts[4] = "{id}"
                    endpoint = "/".join(parts)

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code,
            ).inc()

            REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(
                duration_seconds
            )

        # Log request
        log_request(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
            user_agent=request.headers.get("User-Agent"),
            remote_addr=request.client.host if request.client else None,
        )

        return response

    # Global exception handlers
    @app.exception_handler(NovaEvalAPIError)
    async def novaeval_exception_handler(request: Request, exc: NovaEvalAPIError):
        """Handle custom NovaEval API exceptions."""
        log_error(exc, context={"request_path": str(request.url.path)})

        return JSONResponse(
            status_code=400,
            content={
                "error": exc.message,
                "error_type": exc.error_code,
                "details": exc.details,
            },
        )

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors."""
        log_error(exc, context={"request_path": str(request.url.path)})

        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation failed",
                "error_type": "ValidationError",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler_custom(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with logging."""
        if exc.status_code >= 500:
            log_error(exc, context={"request_path": str(request.url.path)})

        return await http_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all other unhandled exceptions."""
        log_error(exc, context={"request_path": str(request.url.path)})

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_type": "InternalServerError",
                "details": {},
            },
        )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    # Add manual metrics endpoint (optional)
    if PROMETHEUS_AVAILABLE:

        @app.get("/metrics", include_in_schema=False)
        async def metrics_endpoint():
            """Prometheus metrics endpoint."""
            from fastapi.responses import Response

            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


# Create the app instance
app = create_app()


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint for container readiness and liveness probes.

    Returns:
        Dict[str, Any]: Health status information
    """
    return {"status": "healthy", "service": "novaeval-api", "version": "1.0.0"}


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """
    Root endpoint with basic API information.

    Returns:
        Dict[str, str]: API welcome message
    """
    return {"message": "Welcome to NovaEval API", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
