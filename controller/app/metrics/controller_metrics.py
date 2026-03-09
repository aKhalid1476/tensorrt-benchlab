"""Prometheus metrics for TensorRT BenchLab Controller."""
from prometheus_client import Counter, Histogram, Gauge, Info

# ==============================================================================
# Run Metrics
# ==============================================================================

runs_total = Counter(
    "benchlab_controller_runs_total",
    "Total number of benchmark runs created",
    labelnames=["model", "status"],
)

runs_by_status = Gauge(
    "benchlab_controller_runs_by_status",
    "Number of runs in each status",
    labelnames=["status"],
)

run_duration_seconds = Histogram(
    "benchlab_controller_run_duration_seconds",
    "Duration of benchmark runs from creation to completion",
    labelnames=["model", "status"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600],
)

# ==============================================================================
# Runner Communication Metrics
# ==============================================================================

runner_request_duration_seconds = Histogram(
    "benchlab_controller_runner_request_duration_seconds",
    "Duration of runner API requests",
    labelnames=["engine", "status"],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
)

runner_request_retries_total = Counter(
    "benchlab_controller_runner_request_retries_total",
    "Total number of runner request retries",
    labelnames=["engine", "attempt"],
)

# ==============================================================================
# Active Run Tracking
# ==============================================================================

active_runs = Gauge(
    "benchlab_controller_active_runs",
    "Number of currently active (running) benchmark runs",
)

# ==============================================================================
# Controller Info
# ==============================================================================

controller_info = Info(
    "benchlab_controller",
    "Controller version and configuration information",
)

controller_info.info(
    {
        "version": "0.1.0",
        "service": "tensorrt-benchlab-controller",
        "python_version": "3.11",
    }
)
