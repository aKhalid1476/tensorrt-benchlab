"""Prometheus metrics for TensorRT BenchLab.

This module defines Prometheus metrics for monitoring:
- API request latencies (histograms)
- Benchmark run durations
- GPU utilization and metrics
- System health
"""
import logging
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)

# ============================================================================
# API Request Metrics
# ============================================================================

# Request latency histogram
api_request_duration_seconds = Histogram(
    "benchlab_api_request_duration_seconds",
    "API request duration in seconds",
    labelnames=["method", "endpoint", "status"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Request counter
api_requests_total = Counter(
    "benchlab_api_requests_total",
    "Total number of API requests",
    labelnames=["method", "endpoint", "status"],
)

# ============================================================================
# Benchmark Run Metrics
# ============================================================================

# Benchmark run duration
benchmark_run_duration_seconds = Histogram(
    "benchlab_benchmark_run_duration_seconds",
    "Benchmark run duration in seconds",
    labelnames=["model", "engine", "status"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0),
)

# Active benchmark runs
benchmark_runs_active = Gauge(
    "benchlab_benchmark_runs_active",
    "Number of currently active benchmark runs",
)

# Benchmark runs counter
benchmark_runs_total = Counter(
    "benchlab_benchmark_runs_total",
    "Total number of benchmark runs",
    labelnames=["model", "engine", "status"],
)

# Benchmark iteration latency (per batch)
benchmark_iteration_latency_ms = Histogram(
    "benchlab_benchmark_iteration_latency_ms",
    "Benchmark iteration latency in milliseconds",
    labelnames=["model", "engine", "batch_size"],
    buckets=(1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0),
)

# ============================================================================
# GPU Metrics
# ============================================================================

# GPU utilization percentage
gpu_utilization_percent = Gauge(
    "benchlab_gpu_utilization_percent",
    "GPU utilization percentage",
    labelnames=["device"],
)

# GPU memory used (MB)
gpu_memory_used_mb = Gauge(
    "benchlab_gpu_memory_used_mb",
    "GPU memory used in MB",
    labelnames=["device"],
)

# GPU memory total (MB)
gpu_memory_total_mb = Gauge(
    "benchlab_gpu_memory_total_mb",
    "GPU memory total in MB",
    labelnames=["device"],
)

# GPU temperature (Celsius)
gpu_temperature_celsius = Gauge(
    "benchlab_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    labelnames=["device"],
)

# GPU power usage (Watts)
gpu_power_usage_watts = Gauge(
    "benchlab_gpu_power_usage_watts",
    "GPU power usage in Watts",
    labelnames=["device"],
)

# ============================================================================
# System Info
# ============================================================================

# System information
system_info = Info(
    "benchlab_system",
    "System information",
)


def update_gpu_metrics(
    device_name: str,
    utilization: float,
    memory_used: float,
    memory_total: float,
    temperature: Optional[float] = None,
    power: Optional[float] = None,
) -> None:
    """
    Update GPU metrics from telemetry sample.

    Args:
        device_name: GPU device name
        utilization: GPU utilization percentage (0-100)
        memory_used: Memory used in MB
        memory_total: Total memory in MB
        temperature: Temperature in Celsius (optional)
        power: Power usage in Watts (optional)
    """
    try:
        gpu_utilization_percent.labels(device=device_name).set(utilization)
        gpu_memory_used_mb.labels(device=device_name).set(memory_used)
        gpu_memory_total_mb.labels(device=device_name).set(memory_total)

        if temperature is not None:
            gpu_temperature_celsius.labels(device=device_name).set(temperature)

        if power is not None:
            gpu_power_usage_watts.labels(device=device_name).set(power)

    except Exception as e:
        logger.error(f"event=update_gpu_metrics_failed error={e}")


def set_system_info(info: dict) -> None:
    """
    Set system information.

    Args:
        info: Dictionary with system information
    """
    try:
        system_info.info(info)
    except Exception as e:
        logger.error(f"event=set_system_info_failed error={e}")


# ============================================================================
# Convenience Functions
# ============================================================================


def record_api_request(method: str, endpoint: str, status: int, duration: float) -> None:
    """
    Record API request metrics.

    Args:
        method: HTTP method
        endpoint: API endpoint path
        status: HTTP status code
        duration: Request duration in seconds
    """
    try:
        api_request_duration_seconds.labels(
            method=method, endpoint=endpoint, status=str(status)
        ).observe(duration)
        api_requests_total.labels(
            method=method, endpoint=endpoint, status=str(status)
        ).inc()
    except Exception as e:
        logger.error(f"event=record_api_request_failed error={e}")


def record_benchmark_run(
    model: str, engine: str, status: str, duration: float
) -> None:
    """
    Record benchmark run metrics.

    Args:
        model: Model name
        engine: Engine type
        status: Run status (completed, failed)
        duration: Run duration in seconds
    """
    try:
        benchmark_run_duration_seconds.labels(
            model=model, engine=engine, status=status
        ).observe(duration)
        benchmark_runs_total.labels(
            model=model, engine=engine, status=status
        ).inc()
    except Exception as e:
        logger.error(f"event=record_benchmark_run_failed error={e}")


def record_iteration_latency(
    model: str, engine: str, batch_size: int, latency_ms: float
) -> None:
    """
    Record benchmark iteration latency.

    Args:
        model: Model name
        engine: Engine type
        batch_size: Batch size
        latency_ms: Latency in milliseconds
    """
    try:
        benchmark_iteration_latency_ms.labels(
            model=model, engine=engine, batch_size=str(batch_size)
        ).observe(latency_ms)
    except Exception as e:
        logger.error(f"event=record_iteration_latency_failed error={e}")
