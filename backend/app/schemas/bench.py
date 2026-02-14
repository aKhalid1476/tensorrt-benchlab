"""Pydantic schemas for benchmark API contracts."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EngineType(str, Enum):
    """Supported inference engines."""
    PYTORCH_CPU = "pytorch_cpu"
    PYTORCH_GPU = "pytorch_gpu"
    TENSORRT = "tensorrt"


class BenchmarkRequest(BaseModel):
    """Request schema for starting a benchmark run."""
    model_name: str = Field(..., description="Model identifier (e.g., 'resnet50')")
    engine_type: EngineType = Field(..., description="Inference engine to benchmark")
    batch_sizes: list[int] = Field(
        default=[1, 4, 8, 16],
        description="List of batch sizes to test"
    )
    num_iterations: int = Field(default=100, ge=1, description="Number of measured iterations")
    warmup_iterations: int = Field(default=3, ge=0, description="Number of warmup iterations")


class EnvironmentMetadata(BaseModel):
    """Environment and system metadata."""
    gpu_name: Optional[str] = None
    gpu_driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    torch_version: str
    tensorrt_version: Optional[str] = None
    cpu_model: Optional[str] = None
    timestamp: datetime
    git_commit: Optional[str] = None


class BatchStats(BaseModel):
    """Statistics for a single batch size."""
    batch_size: int
    latency_p50_ms: float = Field(..., description="50th percentile latency in milliseconds")
    latency_p95_ms: float = Field(..., description="95th percentile latency in milliseconds")
    latency_mean_ms: float = Field(..., description="Mean latency in milliseconds")
    latency_stddev_ms: float = Field(..., description="Standard deviation of latency")
    throughput_req_per_sec: float = Field(..., description="Requests per second")


class BenchmarkResult(BaseModel):
    """Complete benchmark result."""
    run_id: str = Field(..., description="Unique identifier for this benchmark run")
    model_name: str
    engine_type: EngineType
    environment: EnvironmentMetadata
    results: list[BatchStats] = Field(..., description="Results per batch size")
    status: str = Field(default="completed", description="Status: pending, running, completed, failed")
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class BenchmarkRunResponse(BaseModel):
    """Response when starting a benchmark run."""
    run_id: str
    status: str
    message: str


class TelemetrySample(BaseModel):
    """Single telemetry sample from GPU."""
    timestamp: datetime
    gpu_utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_celsius: Optional[float] = None
    power_usage_watts: Optional[float] = None


class TelemetryResponse(BaseModel):
    """Response for telemetry endpoint."""
    samples: list[TelemetrySample]
    device_name: str
