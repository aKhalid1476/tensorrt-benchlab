"""Prometheus metrics for TensorRT BenchLab Runner."""
from prometheus_client import Counter, Histogram, Gauge, Info

# ==============================================================================
# Benchmark Execution Metrics
# ==============================================================================

benchmark_executions_total = Counter(
    "benchlab_runner_benchmark_executions_total",
    "Total number of benchmark executions",
    labelnames=["model", "engine", "status"],
)

benchmark_duration_seconds = Histogram(
    "benchlab_runner_benchmark_duration_seconds",
    "Total duration of benchmark execution (including warmup)",
    labelnames=["model", "engine"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

# ==============================================================================
# Inference Performance Metrics
# ==============================================================================

inference_latency_seconds = Histogram(
    "benchlab_runner_inference_latency_seconds",
    "Inference latency per batch",
    labelnames=["model", "engine", "batch_size"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

model_load_duration_seconds = Histogram(
    "benchlab_runner_model_load_duration_seconds",
    "Duration to load model",
    labelnames=["model", "engine"],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60],
)

engine_build_duration_seconds = Histogram(
    "benchlab_runner_engine_build_duration_seconds",
    "Duration to build TensorRT engine",
    labelnames=["model", "batch_size", "precision"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

# ==============================================================================
# GPU Telemetry Metrics
# ==============================================================================

gpu_utilization_percent = Gauge(
    "benchlab_runner_gpu_utilization_percent",
    "Current GPU utilization percentage",
)

gpu_memory_used_bytes = Gauge(
    "benchlab_runner_gpu_memory_used_bytes",
    "Current GPU memory used in bytes",
)

gpu_temperature_celsius = Gauge(
    "benchlab_runner_gpu_temperature_celsius",
    "Current GPU temperature in Celsius",
)

gpu_power_usage_watts = Gauge(
    "benchlab_runner_gpu_power_usage_watts",
    "Current GPU power usage in watts",
)

# ==============================================================================
# Runner Info
# ==============================================================================

runner_info = Info(
    "benchlab_runner",
    "Runner version and configuration information",
)

runner_info.info(
    {
        "version": "0.1.0",
        "service": "tensorrt-benchlab-runner",
    }
)
