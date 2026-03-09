"""FastAPI application entry point."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from .api import routes_bench, routes_metrics
from .api.routes_metrics import get_sampler, update_prometheus_gpu_metrics
from .middleware.metrics import MetricsMiddleware
from .telemetry.prometheus_metrics import set_system_info
from .utils.env import get_environment_metadata
from .utils.logging_config import setup_logging

# Setup structured logging
log_level = os.getenv("BENCHLAB_LOG_LEVEL", "INFO")
setup_logging(log_level)
logger = logging.getLogger(__name__)

# Background task for Prometheus metrics update
prometheus_update_task: Optional[asyncio.Task] = None


async def update_prometheus_metrics_loop() -> None:
    """
    Background task to periodically update Prometheus GPU metrics.

    Updates Prometheus metrics every 1 second from the latest NVML sample.
    """
    logger.info("event=prometheus_update_start")

    while True:
        try:
            update_prometheus_gpu_metrics()
            await asyncio.sleep(1.0)  # Update every 1 second
        except asyncio.CancelledError:
            logger.info("event=prometheus_update_cancelled")
            break
        except Exception as e:
            logger.error(f"event=prometheus_update_error error={e}", exc_info=True)
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown of background tasks:
    - NVML GPU telemetry sampling (200ms interval)
    - Prometheus metrics update (1s interval)
    """
    global prometheus_update_task

    logger.info("service=benchlab event=startup version=0.1.0")

    # Get sampler instance
    sampler = get_sampler()

    # Set system info in Prometheus
    try:
        env = get_environment_metadata()
        system_info = {
            "gpu_name": env.gpu_name or "Unknown",
            "cuda_version": env.cuda_version or "Unknown",
            "torch_version": env.torch_version,
        }
        set_system_info(system_info)
        logger.info(f"event=system_info_set info={system_info}")
    except Exception as e:
        logger.error(f"event=system_info_failed error={e}")

    # Start NVML background sampling
    sampler.start_sampling()

    # Start Prometheus metrics update task
    prometheus_update_task = asyncio.create_task(update_prometheus_metrics_loop())

    logger.info("event=startup_complete")

    yield

    # Shutdown
    logger.info("service=benchlab event=shutdown_start")

    # Stop Prometheus update task
    if prometheus_update_task is not None:
        prometheus_update_task.cancel()
        try:
            await asyncio.wait_for(prometheus_update_task, timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("event=prometheus_update_shutdown_timeout")
        except asyncio.CancelledError:
            pass

    # Stop NVML sampling
    await sampler.stop_sampling()

    logger.info("service=benchlab event=shutdown_complete")


app = FastAPI(
    title="TensorRT BenchLab API",
    version="0.1.0",
    description="Production-quality inference benchmarking across PyTorch and TensorRT",
    lifespan=lifespan,
)

# Metrics middleware (must be added before CORS)
app.add_middleware(MetricsMiddleware)

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
