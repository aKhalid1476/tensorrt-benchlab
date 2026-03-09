"""Pydantic schemas for benchmark API contracts.

These schemas are shared between controller and runner services.
Schema Version: 1.0.0
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Schema version - increment when breaking changes occur
SCHEMA_VERSION = "1.0.0"


class EngineType(str, Enum):
    """Supported inference engines."""

    PYTORCH_CPU = "pytorch_cpu"
    PYTORCH_CUDA = "pytorch_cuda"
    TENSORRT = "tensorrt"


class RunStatus(str, Enum):
    """Benchmark run status."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# Run Configuration
# ============================================================================


class RunCreateRequest(BaseModel):
    """Request to create a new benchmark run."""

    runner_url: str = Field(..., description="URL of runner service")
    model_name: str = Field(..., description="Model identifier (e.g., 'resnet50')")
    engines: list[EngineType] = Field(
        ..., description="List of engines to benchmark"
    )
    batch_sizes: list[int] = Field(
        default=[1, 4, 8, 16], description="List of batch sizes to test"
    )
    num_iterations: int = Field(
        default=50, ge=1, description="Number of measured iterations"
    )
    warmup_iterations: int = Field(
        default=10, ge=0, description="Number of warmup iterations"
    )
    client_run_key: Optional[str] = Field(
        None, description="Optional idempotency key"
    )


# ============================================================================
# Results
# ============================================================================


class TimingBreakdown(BaseModel):
    """Timing breakdown for benchmark execution."""

    total_duration_sec: float
    model_load_sec: float
    warmup_duration_sec: float
    measurement_duration_sec: float
    preprocessing_ms: Optional[float] = None
    forward_ms: Optional[float] = None
    postprocessing_ms: Optional[float] = None


class EngineResult(BaseModel):
    """Results for a single engine and batch size."""

    engine_name: str = Field(..., description="Engine that ran this benchmark")
    batch_size: int
    latency_p50_ms: float = Field(
        ..., description="50th percentile latency in milliseconds"
    )
    latency_p95_ms: float = Field(
        ..., description="95th percentile latency in milliseconds"
    )
    latency_mean_ms: float = Field(..., description="Mean latency in milliseconds")
    latency_stddev_ms: float = Field(..., description="Standard deviation of latency")
    throughput_req_per_sec: float = Field(..., description="Requests per second")
    timing_breakdown: Optional[TimingBreakdown] = None


class EnvironmentMetadata(BaseModel):
    """Environment and system metadata."""

    gpu_name: Optional[str] = None
    gpu_driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    torch_version: str
    tensorrt_version: Optional[str] = None
    python_version: Optional[str] = None
    cpu_model: Optional[str] = None
    timestamp: datetime
    git_commit: Optional[str] = None
    sanity_check_passed: Optional[bool] = Field(
        None, description="Whether cross-engine output sanity check passed"
    )


# ============================================================================
# Telemetry
# ============================================================================


class TelemetrySample(BaseModel):
    """Single telemetry sample from GPU."""

    t_ms: float = Field(..., description="Time in milliseconds relative to run start")
    gpu_utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_celsius: Optional[float] = None
    power_usage_watts: Optional[float] = None


class TelemetryResponse(BaseModel):
    """Telemetry data for a run."""

    samples: list[TelemetrySample]
    device_name: str
    run_id: Optional[str] = None


# ============================================================================
# Run Record (Controller-managed)
# ============================================================================


class RunRecord(BaseModel):
    """Complete run record stored by controller."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str
    status: RunStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Configuration
    runner_url: str
    model_name: str
    engines: list[EngineType]
    batch_sizes: list[int]
    num_iterations: int
    warmup_iterations: int
    client_run_key: Optional[str] = None

    # Results
    environment: Optional[EnvironmentMetadata] = None
    results: list[EngineResult] = Field(default_factory=list)
    telemetry: Optional[TelemetryResponse] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None


class RunCreateResponse(BaseModel):
    """Response when creating a run."""

    run_id: str
    status: RunStatus
    message: str


class RunListResponse(BaseModel):
    """Response for listing runs."""

    runs: list[RunRecord]
    total: int


# ============================================================================
# Runner Communication (Controller <-> Runner)
# ============================================================================


class RunnerExecuteRequest(BaseModel):
    """Request to runner to execute a benchmark for a single engine."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str
    model_name: str
    engine_type: EngineType
    batch_sizes: list[int]
    num_iterations: int
    warmup_iterations: int


class RunnerExecuteResponse(BaseModel):
    """Response from runner after executing benchmark."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str
    status: str = Field(..., description="succeeded or failed")
    environment: EnvironmentMetadata
    results: list[EngineResult] = Field(default_factory=list)
    telemetry: TelemetryResponse
    timing: TimingBreakdown
    error_message: Optional[str] = None
    error_stack: Optional[str] = None


class RunnerVersionResponse(BaseModel):
    """Runner version and environment information."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    runner_version: str
    torch_version: str
    cuda_version: Optional[str] = None
    tensorrt_version: Optional[str] = None
    python_version: str
    gpu_name: Optional[str] = None
    git_commit: Optional[str] = None


# ============================================================================
# Legacy Compatibility (for existing code)
# ============================================================================


# Aliases for backward compatibility
BatchStats = EngineResult
BenchmarkRequest = RunCreateRequest
BenchmarkResult = RunRecord
BenchmarkRunResponse = RunCreateResponse
ExecuteRequest = RunnerExecuteRequest
ExecuteResponse = RunnerExecuteResponse
