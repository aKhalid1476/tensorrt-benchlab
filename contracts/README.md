# TensorRT BenchLab Contracts

Shared Pydantic schemas between controller and runner services.

**Schema Version:** 1.0.0

## Installation

```bash
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all contract tests
pytest

# With verbose output
pytest -v

# Run specific test
pytest tests/test_contracts.py::test_run_record_complete -v
```

## Schemas

### Core Types
- `EngineType` - Enum of supported engines
- `RunStatus` - Enum of run statuses (queued/running/succeeded/failed/cancelled)

### Run Management
- `RunCreateRequest` - Create a new benchmark run
- `RunCreateResponse` - Response when creating run
- `RunRecord` - Complete run record with results
- `RunListResponse` - List of runs

### Results
- `EngineResult` - Results for single engine/batch combination
- `EnvironmentMetadata` - System environment information
- `TimingBreakdown` - Detailed timing information

### Telemetry
- `TelemetrySample` - Single GPU telemetry sample
- `TelemetryResponse` - Collection of telemetry samples

### Runner Communication
- `RunnerExecuteRequest` - Request to runner to execute benchmark
- `RunnerExecuteResponse` - Runner response with results
- `RunnerVersionResponse` - Runner version information

## Schema Versioning

All schemas include `schema_version` field (default: "1.0.0").

Increment version when making breaking changes:
- **Major:** Breaking changes (e.g., 1.0.0 → 2.0.0)
- **Minor:** New fields (backward compatible) (e.g., 1.0.0 → 1.1.0)
- **Patch:** Bug fixes/clarifications (e.g., 1.0.0 → 1.0.1)

## Usage Example

```python
from contracts import (
    SCHEMA_VERSION,
    RunCreateRequest,
    EngineType,
    RunStatus,
)

# Create a run request
request = RunCreateRequest(
    runner_url="http://runner:8001",
    model_name="resnet50",
    engines=[EngineType.PYTORCH_CUDA, EngineType.TENSORRT],
    batch_sizes=[1, 4, 8, 16],
)

# Serialize to JSON
json_data = request.model_dump_json()

# Deserialize from JSON
restored = RunCreateRequest.model_validate_json(json_data)
```

## Development

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=contracts --cov-report=html
```
