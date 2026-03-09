"""Tests for contract schemas serialization/deserialization."""
from datetime import datetime, timezone

import pytest
from contracts import (
    SCHEMA_VERSION,
    EngineResult,
    EngineType,
    EnvironmentMetadata,
    RunCreateRequest,
    RunRecord,
    RunStatus,
    RunnerExecuteRequest,
    RunnerExecuteResponse,
    RunnerVersionResponse,
    TelemetryResponse,
    TelemetrySample,
    TimingBreakdown,
)


def test_schema_version_present():
    """Test that schema version is defined."""
    assert SCHEMA_VERSION is not None
    assert isinstance(SCHEMA_VERSION, str)
    assert len(SCHEMA_VERSION) > 0


def test_run_create_request_serialization():
    """Test RunCreateRequest can be serialized and deserialized."""
    request = RunCreateRequest(
        runner_url="http://runner:8001",
        model_name="resnet50",
        engines=[EngineType.PYTORCH_CPU, EngineType.TENSORRT],
        batch_sizes=[1, 4, 8],
        num_iterations=50,
        warmup_iterations=10,
        client_run_key="test-key-123",
    )

    # Serialize to dict
    data = request.model_dump()

    # Deserialize from dict
    restored = RunCreateRequest(**data)

    assert restored.runner_url == request.runner_url
    assert restored.model_name == request.model_name
    assert restored.engines == request.engines
    assert restored.batch_sizes == request.batch_sizes
    assert restored.num_iterations == request.num_iterations
    assert restored.warmup_iterations == request.warmup_iterations
    assert restored.client_run_key == request.client_run_key


def test_run_create_request_json_serialization():
    """Test RunCreateRequest JSON serialization."""
    request = RunCreateRequest(
        runner_url="http://runner:8001",
        model_name="resnet50",
        engines=[EngineType.PYTORCH_CUDA],
        batch_sizes=[1, 4],
    )

    # Serialize to JSON
    json_str = request.model_dump_json()

    # Deserialize from JSON
    restored = RunCreateRequest.model_validate_json(json_str)

    assert restored.runner_url == request.runner_url
    assert restored.engines == request.engines


def test_engine_result_complete():
    """Test EngineResult with all fields."""
    timing = TimingBreakdown(
        total_duration_sec=10.5,
        model_load_sec=1.2,
        warmup_duration_sec=0.8,
        measurement_duration_sec=8.5,
        preprocessing_ms=2.5,
        forward_ms=5.0,
        postprocessing_ms=1.5,
    )

    result = EngineResult(
        engine_name="pytorch_cuda",
        batch_size=4,
        latency_p50_ms=5.2,
        latency_p95_ms=6.8,
        latency_mean_ms=5.5,
        latency_stddev_ms=0.4,
        throughput_req_per_sec=180.5,
        timing_breakdown=timing,
    )

    # Serialize and deserialize
    data = result.model_dump()
    restored = EngineResult(**data)

    assert restored.engine_name == result.engine_name
    assert restored.batch_size == result.batch_size
    assert restored.latency_p50_ms == result.latency_p50_ms
    assert restored.timing_breakdown is not None
    assert restored.timing_breakdown.forward_ms == 5.0


def test_environment_metadata():
    """Test EnvironmentMetadata serialization."""
    env = EnvironmentMetadata(
        gpu_name="NVIDIA A100",
        gpu_driver_version="525.105.17",
        cuda_version="12.1",
        torch_version="2.1.0",
        tensorrt_version="8.6.1",
        python_version="3.11.5",
        cpu_model="AMD EPYC 7763",
        timestamp=datetime.now(timezone.utc),
        git_commit="abc123def456",
        sanity_check_passed=True,
    )

    data = env.model_dump()
    restored = EnvironmentMetadata(**data)

    assert restored.gpu_name == env.gpu_name
    assert restored.cuda_version == env.cuda_version
    assert restored.sanity_check_passed == True


def test_telemetry_sample():
    """Test TelemetrySample serialization."""
    sample = TelemetrySample(
        t_ms=1234.5,
        gpu_utilization_percent=85.5,
        memory_used_mb=12800.0,
        memory_total_mb=24576.0,
        temperature_celsius=72.0,
        power_usage_watts=320.5,
    )

    data = sample.model_dump()
    restored = TelemetrySample(**data)

    assert restored.t_ms == sample.t_ms
    assert restored.gpu_utilization_percent == sample.gpu_utilization_percent
    assert restored.temperature_celsius == sample.temperature_celsius


def test_run_record_complete():
    """Test complete RunRecord serialization."""
    env = EnvironmentMetadata(
        gpu_name="NVIDIA A100",
        cuda_version="12.1",
        torch_version="2.1.0",
        timestamp=datetime.now(timezone.utc),
        sanity_check_passed=True,
    )

    results = [
        EngineResult(
            engine_name="pytorch_cpu",
            batch_size=1,
            latency_p50_ms=100.0,
            latency_p95_ms=120.0,
            latency_mean_ms=105.0,
            latency_stddev_ms=8.5,
            throughput_req_per_sec=9.5,
        ),
        EngineResult(
            engine_name="tensorrt",
            batch_size=1,
            latency_p50_ms=5.0,
            latency_p95_ms=6.0,
            latency_mean_ms=5.2,
            latency_stddev_ms=0.3,
            throughput_req_per_sec=192.3,
        ),
    ]

    telemetry = TelemetryResponse(
        samples=[
            TelemetrySample(
                t_ms=0.0,
                gpu_utilization_percent=0.0,
                memory_used_mb=1024.0,
                memory_total_mb=24576.0,
            ),
            TelemetrySample(
                t_ms=200.0,
                gpu_utilization_percent=95.0,
                memory_used_mb=12800.0,
                memory_total_mb=24576.0,
                temperature_celsius=75.0,
                power_usage_watts=350.0,
            ),
        ],
        device_name="NVIDIA A100",
        run_id="test-run-123",
    )

    record = RunRecord(
        schema_version=SCHEMA_VERSION,
        run_id="test-run-123",
        status=RunStatus.SUCCEEDED,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        runner_url="http://runner:8001",
        model_name="resnet50",
        engines=[EngineType.PYTORCH_CPU, EngineType.TENSORRT],
        batch_sizes=[1, 4, 8],
        num_iterations=50,
        warmup_iterations=10,
        environment=env,
        results=results,
        telemetry=telemetry,
    )

    # Test serialization
    data = record.model_dump()
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["status"] == "succeeded"
    assert len(data["results"]) == 2

    # Test deserialization
    restored = RunRecord(**data)
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.run_id == record.run_id
    assert restored.status == RunStatus.SUCCEEDED
    assert len(restored.results) == 2
    assert restored.results[0].engine_name == "pytorch_cpu"
    assert restored.telemetry is not None
    assert len(restored.telemetry.samples) == 2


def test_run_record_json_roundtrip():
    """Test RunRecord JSON round-trip."""
    record = RunRecord(
        run_id="test-123",
        status=RunStatus.QUEUED,
        created_at=datetime.now(timezone.utc),
        runner_url="http://runner:8001",
        model_name="resnet50",
        engines=[EngineType.PYTORCH_CUDA],
        batch_sizes=[1],
        num_iterations=10,
        warmup_iterations=2,
    )

    # Serialize to JSON
    json_str = record.model_dump_json()

    # Deserialize from JSON
    restored = RunRecord.model_validate_json(json_str)

    assert restored.run_id == record.run_id
    assert restored.status == RunStatus.QUEUED
    assert restored.schema_version == SCHEMA_VERSION


def test_runner_execute_request():
    """Test RunnerExecuteRequest serialization."""
    request = RunnerExecuteRequest(
        run_id="test-run-456",
        model_name="resnet50",
        engine_type=EngineType.TENSORRT,
        batch_sizes=[1, 4, 8, 16],
        num_iterations=50,
        warmup_iterations=10,
    )

    data = request.model_dump()
    assert data["schema_version"] == SCHEMA_VERSION

    restored = RunnerExecuteRequest(**data)
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.engine_type == EngineType.TENSORRT


def test_runner_execute_response():
    """Test RunnerExecuteResponse serialization."""
    env = EnvironmentMetadata(
        torch_version="2.1.0",
        timestamp=datetime.now(timezone.utc),
    )

    timing = TimingBreakdown(
        total_duration_sec=15.5,
        model_load_sec=2.0,
        warmup_duration_sec=1.5,
        measurement_duration_sec=12.0,
    )

    telemetry = TelemetryResponse(
        samples=[],
        device_name="NVIDIA A100",
    )

    response = RunnerExecuteResponse(
        run_id="test-run-789",
        status="succeeded",
        environment=env,
        results=[],
        telemetry=telemetry,
        timing=timing,
    )

    data = response.model_dump()
    assert data["schema_version"] == SCHEMA_VERSION

    restored = RunnerExecuteResponse(**data)
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.status == "succeeded"
    assert restored.timing.total_duration_sec == 15.5


def test_runner_version_response():
    """Test RunnerVersionResponse serialization."""
    version = RunnerVersionResponse(
        runner_version="0.1.0",
        torch_version="2.1.0",
        cuda_version="12.1",
        tensorrt_version="8.6.1",
        python_version="3.11.5",
        gpu_name="NVIDIA A100",
        git_commit="abc123",
    )

    data = version.model_dump()
    assert data["schema_version"] == SCHEMA_VERSION

    json_str = version.model_dump_json()
    restored = RunnerVersionResponse.model_validate_json(json_str)

    assert restored.schema_version == SCHEMA_VERSION
    assert restored.runner_version == "0.1.0"
    assert restored.git_commit == "abc123"


def test_status_enum_values():
    """Test RunStatus enum has expected values."""
    assert RunStatus.QUEUED.value == "queued"
    assert RunStatus.RUNNING.value == "running"
    assert RunStatus.SUCCEEDED.value == "succeeded"
    assert RunStatus.FAILED.value == "failed"
    assert RunStatus.CANCELLED.value == "cancelled"


def test_engine_type_enum_values():
    """Test EngineType enum has expected values."""
    assert EngineType.PYTORCH_CPU.value == "pytorch_cpu"
    assert EngineType.PYTORCH_CUDA.value == "pytorch_cuda"
    assert EngineType.TENSORRT.value == "tensorrt"
