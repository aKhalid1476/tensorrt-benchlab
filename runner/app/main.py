"""FastAPI runner service for executing benchmarks on GPU host."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import sys

from contracts import (
    EngineResult,
    ExecuteRequest,
    ExecuteResponse,
    RunnerVersionResponse,
    SCHEMA_VERSION,
    TelemetryResponse,
    TimingBreakdown,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import REGISTRY, generate_latest

from .bench.runner import BenchmarkRunner
from .metrics import (
    benchmark_executions_total,
    benchmark_duration_seconds,
    gpu_utilization_percent,
    gpu_memory_used_bytes,
    gpu_temperature_celsius,
    gpu_power_usage_watts,
)
from .telemetry.nvml_sampler import NVMLSampler
from .utils.env import get_environment_metadata
from .utils.logging_config import setup_logging

# Setup structured logging
log_level = os.getenv("BENCHLAB_LOG_LEVEL", "INFO")
setup_logging(log_level)
logger = logging.getLogger(__name__)

# Global sampler
sampler = NVMLSampler()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager.

    Starts/stops GPU telemetry sampling.
    """
    logger.info("service=runner event=startup version=0.1.0")

    # Start NVML sampling
    sampler.start_sampling()

    yield

    # Shutdown
    logger.info("service=runner event=shutdown_start")
    await sampler.stop_sampling()
    logger.info("service=runner event=shutdown_complete")


app = FastAPI(
    title="TensorRT BenchLab Runner",
    version="0.1.0",
    description="GPU-based benchmark execution service",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(
        f"path={request.url.path} method={request.method} error={str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "error": str(exc)}
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "tensorrt-benchlab-runner"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    # Update GPU metrics from sampler
    if sampler.enabled:
        latest_sample = sampler.sample()
        if latest_sample.gpu_utilization_percent is not None:
            gpu_utilization_percent.set(latest_sample.gpu_utilization_percent)
        if latest_sample.memory_used_mb is not None:
            gpu_memory_used_bytes.set(latest_sample.memory_used_mb * 1024 * 1024)
        if latest_sample.temperature_celsius is not None:
            gpu_temperature_celsius.set(latest_sample.temperature_celsius)
        if latest_sample.power_usage_watts is not None:
            gpu_power_usage_watts.set(latest_sample.power_usage_watts)

    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/version", response_model=RunnerVersionResponse)
async def version() -> RunnerVersionResponse:
    """Version information."""
    import torch

    from .utils.env import get_git_commit

    cuda_version = None
    gpu_name = None
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)

    tensorrt_version = None
    try:
        import tensorrt as trt
        tensorrt_version = trt.__version__
    except ImportError:
        pass

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return RunnerVersionResponse(
        runner_version="0.1.0",
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        tensorrt_version=tensorrt_version,
        python_version=python_version,
        gpu_name=gpu_name,
        git_commit=get_git_commit(),
    )


@app.post("/execute", response_model=ExecuteResponse)
async def execute_benchmark(request: ExecuteRequest) -> ExecuteResponse:
    """
    Execute benchmark and return results with telemetry.

    This endpoint runs the benchmark synchronously and returns complete results.
    """
    from .telemetry.run_telemetry import RunTelemetry

    logger.info(
        f"event=execute_request run_id={request.run_id} model={request.model_name} "
        f"engine={request.engine_type}"
    )

    # Start run-scoped telemetry
    run_telemetry = RunTelemetry(sampler, request.run_id)
    await run_telemetry.start()

    try:
        # Run benchmark
        runner = BenchmarkRunner()
        results, timing = await runner.run_benchmark(
            model_name=request.model_name,
            engine_type=request.engine_type,
            batch_sizes=request.batch_sizes,
            num_iterations=request.num_iterations,
            warmup_iterations=request.warmup_iterations,
        )

        # Stop telemetry collection
        await run_telemetry.stop()

        # Get telemetry samples (with relative timestamps)
        telemetry_samples = run_telemetry.get_samples()
        telemetry = TelemetryResponse(
            samples=telemetry_samples,
            device_name=sampler.get_device_name(),
            run_id=request.run_id,
        )

        # Get environment metadata (without sanity check for now)
        env = get_environment_metadata()

        # Record metrics
        benchmark_executions_total.labels(
            model=request.model_name,
            engine=request.engine_type.value,
            status="succeeded",
        ).inc()
        benchmark_duration_seconds.labels(
            model=request.model_name, engine=request.engine_type.value
        ).observe(timing.total_duration_sec)

        logger.info(
            f"event=execute_complete run_id={request.run_id} "
            f"results_count={len(results)} duration_sec={timing.total_duration_sec:.2f} "
            f"telemetry_samples={len(telemetry_samples)}"
        )

        return ExecuteResponse(
            run_id=request.run_id,
            status="succeeded",
            environment=env,
            results=results,
            telemetry=telemetry,
            timing=timing,
        )

    except Exception as e:
        logger.error(
            f"event=execute_failed run_id={request.run_id} error={str(e)}", exc_info=True
        )

        # Record failure metrics
        benchmark_executions_total.labels(
            model=request.model_name,
            engine=request.engine_type.value,
            status="failed",
        ).inc()

        # Stop telemetry
        await run_telemetry.stop()

        # Get whatever telemetry we collected before failure
        telemetry_samples = run_telemetry.get_samples()

        # Return failed response with error and stack trace
        import traceback

        error_stack = traceback.format_exc()

        return ExecuteResponse(
            run_id=request.run_id,
            status="failed",
            environment=get_environment_metadata(),
            results=[],
            telemetry=TelemetryResponse(
                samples=telemetry_samples,
                device_name=sampler.get_device_name(),
                run_id=request.run_id,
            ),
            timing=TimingBreakdown(
                total_duration_sec=0.0,
                model_load_sec=0.0,
                warmup_duration_sec=0.0,
                measurement_duration_sec=0.0,
            ),
            error_message=str(e),
            error_stack=error_stack,
        )
