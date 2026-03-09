# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorRT BenchLab is a distributed benchmarking platform that compares inference performance across PyTorch CPU, PyTorch CUDA, and TensorRT engines. It uses a **controller/runner** split architecture so that the orchestration layer (controller) can run on Mac M2 while only the runner requires an NVIDIA GPU.

**Data flow:** Frontend → Controller (Mac-compatible, port 8000) → Runner (NVIDIA GPU only, port 8001) → Controller → Frontend

## Common Commands

### Installation (run from repo root)
```bash
make install              # Install all packages (contracts, controller, runner)
make install-controller   # Install just contracts + controller
make install-runner       # Install just contracts + runner
```

### Testing
```bash
make test                 # Run all tests
make test-contracts       # Run contracts tests only (no GPU needed)
make test-controller      # Run controller tests only (no GPU needed)
make test-runner          # Run runner tests (requires GPU)

# Single test file
cd contracts && pytest tests/test_contracts.py -v
cd controller && pytest tests/test_something.py::test_name -v
```

### Linting
```bash
make lint                 # Ruff on all packages
cd controller && ruff check app/
cd runner && ruff check app/
```

### Development Servers (run in separate terminals)
```bash
make dev-controller       # Controller on http://localhost:8000
make dev-runner           # Runner on http://localhost:8001 (requires GPU)
```

### Docker
```bash
make build-runner         # Build runner Docker image (needs nvidia-container-toolkit)
make run-runner           # Run runner container with --gpus all
make up                   # Start controller + frontend via docker-compose
make down                 # Stop docker-compose services
make logs                 # Tail all docker-compose logs
```

### Benchmark via API
```bash
make bench                # Quick benchmark (pytorch_cpu, resnet50, batch 1+4)

# Full benchmark
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"runner_url":"http://localhost:8001","model_name":"resnet50",
       "engines":["pytorch_cpu","pytorch_cuda","tensorrt"],
       "batch_sizes":[1,4,8,16],"num_iterations":50}'
```

### Report generation
```bash
# Via API
curl http://localhost:8000/runs/{run_id}/report.md

# Via CLI
benchlab-report --run-id <run_id>
```

## Architecture

### Package structure
- **`contracts/`** — Shared Pydantic schemas (v1.0.0). Installed as an editable package. Both controller and runner `import from contracts`. Do not add GPU/CUDA dependencies here.
- **`controller/`** — FastAPI orchestration service. Mac M2 compatible. No torch/CUDA/NVML. Persists runs to SQLite via SQLModel.
- **`runner/`** — FastAPI execution service. Requires NVIDIA GPU. Houses inference engines, NVML telemetry, benchmark methodology.
- **`frontend/`** — React + TypeScript + Vite dashboard (port 5173). Recharts for visualization.

### Key design patterns

**Engine abstraction** — All inference engines implement `InferenceEngine` (ABC) in [runner/app/engines/base.py](runner/app/engines/base.py):
- `load_model()` — Load weights, move to device, set `_loaded = True`
- `infer(inputs: torch.Tensor) -> float` — Forward pass, return latency in seconds
- Properties: `name`, `device`

Concrete engines: `TorchCPUEngine`, `TorchCUDAEngine`, `TensorRTEngine`. `BenchmarkRunner` ([runner/app/bench/runner.py](runner/app/bench/runner.py)) instantiates the right engine from a dict keyed by `EngineType`.

**Run lifecycle** (controller-managed):
1. `POST /runs` creates `RunDB` in SQLite with `status=QUEUED`, returns `run_id`
2. FastAPI `BackgroundTasks` calls `execute_run_on_runner()`
3. Controller iterates engines, calling `POST /execute` on runner for each (with 3-attempt exponential backoff)
4. Results aggregated, status set to `SUCCEEDED` or `FAILED`
5. Client polls `GET /runs/{run_id}`

**Benchmark methodology** ([runner/app/bench/methodology.py](runner/app/bench/methodology.py)):
- Fixed random seed = 42
- Warmup iterations excluded
- Stats: p50, p95, mean, stddev (sample), throughput = `batch_size / mean_latency_sec`
- `calculate_batch_statistics()` takes latencies in **seconds**, returns `BatchStats` in **ms**

**Telemetry** — `NVMLSampler` runs a background thread sampling every 200ms. `RunTelemetry` captures run-scoped samples with `t_ms` relative to run start. On NVML unavailable, gracefully returns empty telemetry.

### Schemas (contracts/contracts/schemas.py)
Key types: `RunCreateRequest`, `RunRecord`, `RunStatus`, `EngineType`, `EngineResult`, `TimingBreakdown`, `TelemetrySample`, `RunnerExecuteRequest`, `RunnerExecuteResponse`. Legacy aliases (`ExecuteRequest`, `ExecuteResponse`, `BatchStats`) exist for backward compatibility.

### Controller database
`RunDB` SQLModel stores config and JSON-serialized blobs: `engines_json`, `batch_sizes_json`, `results_json`, `environment_json`, `telemetry_json`. The DB file is created automatically on startup via `init_db()`.

### Observability
Both services expose `GET /metrics` (Prometheus). Controller tracks: runs_total, runs_by_status, run_duration_seconds, runner_request_duration, retries, active_runs. Runner tracks: benchmark executions, duration, inference latency histograms, GPU util/mem/temp/power.

### Logging convention
Use `key=value` structured format: `f"event=foo run_id={run_id} engine={engine}"`. Log level controlled by `LOG_LEVEL` (controller) or `BENCHLAB_LOG_LEVEL` (runner) env vars.

## Hardware Constraints

- **Local dev machine is Apple Silicon M2** — no CUDA/TensorRT/NVML available locally
- `controller` and `contracts` must remain CUDA-free and testable locally
- `runner` is the only service that imports torch CUDA, tensorrt, or pynvml — always runs on a remote GPU host

## Adding a New Inference Engine

1. Subclass `InferenceEngine` in `runner/app/engines/`
2. Add a value to `EngineType` enum in `contracts/contracts/schemas.py`
3. Register it in `BenchmarkRunner.engines` dict in `runner/app/bench/runner.py`

## Adding a New Model

Add a new workload in `runner/app/bench/workloads.py` and ensure `prepare_fixed_inputs()` handles the model's input shape. The engine's `input_shape` attribute determines tensor dimensions.
