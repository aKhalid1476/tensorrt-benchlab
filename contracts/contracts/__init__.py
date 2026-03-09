"""Shared contracts between controller and runner services.

Schema Version: 1.0.0
"""
from .schemas import (
    SCHEMA_VERSION,
    EngineResult,
    EngineType,
    EnvironmentMetadata,
    RunCreateRequest,
    RunCreateResponse,
    RunListResponse,
    RunRecord,
    RunStatus,
    RunnerExecuteRequest,
    RunnerExecuteResponse,
    RunnerVersionResponse,
    TelemetryResponse,
    TelemetrySample,
    TimingBreakdown,
    # Legacy aliases
    BatchStats,
    BenchmarkRequest,
    BenchmarkResult,
    BenchmarkRunResponse,
    ExecuteRequest,
    ExecuteResponse,
)

__all__ = [
    # Version
    "SCHEMA_VERSION",
    # Core types
    "EngineType",
    "RunStatus",
    # Run management
    "RunCreateRequest",
    "RunCreateResponse",
    "RunListResponse",
    "RunRecord",
    # Results
    "EngineResult",
    "EnvironmentMetadata",
    "TimingBreakdown",
    # Telemetry
    "TelemetryResponse",
    "TelemetrySample",
    # Runner communication
    "RunnerExecuteRequest",
    "RunnerExecuteResponse",
    "RunnerVersionResponse",
    # Legacy aliases
    "BatchStats",
    "BenchmarkRequest",
    "BenchmarkResult",
    "BenchmarkRunResponse",
    "ExecuteRequest",
    "ExecuteResponse",
]

__version__ = SCHEMA_VERSION
