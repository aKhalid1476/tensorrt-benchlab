"""FastAPI application entry point."""
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from .api import routes_bench, routes_metrics
from .utils.logging_config import setup_logging

# Setup structured logging
log_level = os.getenv("BENCHLAB_LOG_LEVEL", "INFO")
setup_logging(log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager."""
    logger.info("service=benchlab event=startup version=0.1.0")
    yield
    logger.info("service=benchlab event=shutdown")


app = FastAPI(
    title="TensorRT BenchLab API",
    version="0.1.0",
    description="Production-quality inference benchmarking across PyTorch and TensorRT",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes_bench.router, prefix="/bench", tags=["benchmark"])
app.include_router(routes_metrics.router, prefix="/telemetry", tags=["telemetry"])

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(
        f"path={request.url.path} method={request.method} error={str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    logger.debug("event=health_check")
    return {"status": "healthy", "service": "tensorrt-benchlab"}
