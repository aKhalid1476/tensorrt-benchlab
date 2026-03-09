# ⚡ TensorRT BenchLab

Production-quality distributed benchmarking platform for comparing inference performance across PyTorch CPU, PyTorch GPU, and TensorRT engines.

## Architecture

**Distributed Design:**
- **Controller** (Mac M2 compatible): Orchestrates benchmark runs, SQLite persistence, API gateway
- **Runner** (NVIDIA GPU required): Executes benchmarks on GPU hardware, collects telemetry
- **Contracts**: Shared Pydantic schemas with versioning (v1.0.0)

This architecture allows the controller to run on any machine (Mac, Linux, cloud) while the runner executes on expensive GPU instances only when needed.

## Features

- **Multiple Inference Engines**: PyTorch CPU, PyTorch CUDA, TensorRT (FP16/FP32)
- **Production TensorRT**: Real ONNX export, engine building with optimization profiles, disk caching
- **Reproducible Methodology**: Fixed seeds, warmup exclusion, p50/p95/throughput statistics
- **Run-Scoped Telemetry**: GPU util/mem/power/temp sampled at 200ms with relative timestamps
- **Correctness Sanity Checks**: Cross-engine validation of top-1 class predictions
- **Full Observability**: Environment metadata, timing breakdowns, structured logging
- **Docker-Ready**: Runner Dockerfile with NVIDIA CUDA base, docker-compose for controller+frontend

## Quick Start

### Prerequisites

**Controller (Required):**
- Python 3.11+
- Any OS (Mac M2, Linux, Windows)

**Runner (Required for GPU benchmarks):**
- NVIDIA GPU with CUDA 12.1+ drivers
- Docker with nvidia-container-toolkit
- Linux (GPU host)

### Step 1: Start Runner (on GPU host)

```bash
# Build runner Docker image
docker build -t tensorrt-benchlab-runner -f runner/Dockerfile .

# Run with GPU access
docker run --gpus all -p 8001:8001 \
  -v $(pwd)/cache:/app/cache \
  tensorrt-benchlab-runner
```

Verify runner is healthy:
```bash
curl http://<gpu-host>:8001/health
curl http://<gpu-host>:8001/version | jq .
```

### Step 2: Start Controller (on local machine)

```bash
# Install contracts package
cd contracts && pip install -e . && cd ..

# Install and run controller
cd controller
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Access:
- Controller API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Step 3: Create a Benchmark Run

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "runner_url": "http://<gpu-host>:8001",
    "model_name": "resnet50",
    "engines": ["pytorch_cuda", "tensorrt"],
    "batch_sizes": [1, 4, 8, 16],
    "num_iterations": 50,
    "warmup_iterations": 10
  }' | jq .
```

### Step 4: Get Results

```bash
# Get run status and results (replace {run_id})
curl http://localhost:8000/runs/{run_id} | jq .

# List all runs
curl http://localhost:8000/runs | jq .
```

## Project Structure

```
tensorrt-benchlab/
├── contracts/            # Shared Pydantic schemas (v1.0.0)
│   ├── contracts/
│   │   ├── __init__.py   # All schema exports
│   │   └── schemas.py    # Schema definitions
│   ├── tests/            # Contract tests
│   └── pyproject.toml
├── controller/           # Orchestration service (Mac M2 compatible)
│   ├── app/
│   │   ├── db/           # SQLModel + SQLite persistence
│   │   ├── api/          # Run management endpoints
│   │   └── main.py       # FastAPI app
│   └── pyproject.toml
├── runner/               # GPU execution service (NVIDIA required)
│   ├── app/
│   │   ├── engines/      # torch_cpu, torch_cuda, tensorrt
│   │   ├── bench/        # Benchmark runner
│   │   ├── telemetry/    # NVML sampling + run telemetry
│   │   ├── utils/        # env metadata, timing, sanity checks
│   │   └── main.py       # FastAPI app
│   ├── Dockerfile        # NVIDIA CUDA 12.1 base
│   └── pyproject.toml
├── frontend/             # React + TypeScript (TODO)
├── docs/                 # Documentation (TODO)
│   ├── METHODOLOGY.md
│   └── DEPLOY_RUNNER.md
├── scripts/              # Utilities (TODO)
│   └── demo.sh
├── PROGRESS.md           # Implementation status
└── CLAUDE.md             # AI agent instructions
```

## API Endpoints

### Controller API (Port 8000)

**Run Management:**
- `POST /runs` - Create new benchmark run (returns run_id)
  - Body: `RunCreateRequest` (runner_url, model_name, engines, batch_sizes, etc.)
  - Returns: `RunCreateResponse` (run_id, status=queued)
- `GET /runs` - List all runs
  - Returns: `RunListResponse` (array of RunRecord)
- `GET /runs/{run_id}` - Get run details
  - Returns: `RunRecord` (status, results, telemetry, timing, environment)
- `GET /health` - Health check

### Runner API (Port 8001)

**Execution (Internal - called by controller):**
- `POST /execute` - Execute benchmark
  - Body: `ExecuteRequest` (run_id, model_name, engine_type, batch_sizes, iterations)
  - Returns: `ExecuteResponse` (results, telemetry, timing, environment)
- `GET /version` - Runner version and environment info
  - Returns: `RunnerVersionResponse` (torch, CUDA, TensorRT versions, GPU name)
- `GET /health` - Health check

## Benchmark Methodology

- **Warmup**: Configurable warmup iterations (default: 10) excluded from measurements
- **Measurement**: N iterations (default: 50) with precise timing
- **Batch Sizes**: [1, 4, 8, 16] by default, configurable
- **Statistics**: p50, p95, mean, stddev latency + throughput (img/s)
- **Reproducibility**: Fixed random seed (42), fixed input tensors
- **CUDA Synchronization**: `torch.cuda.synchronize()` before/after forward pass for accurate GPU timing
- **Timing Breakdown**: Separate preprocessing, forward pass, and postprocessing measurements
- **Sanity Checks**: Cross-engine validation - compares top-1 class predictions across all engines

## TensorRT Implementation

- **ONNX Export**: Dynamic batch dimension support, opset 14
- **Engine Building**: Optimization profiles for each batch size
- **Precision**: FP16 by default with automatic FP32 fallback if unsupported
- **Disk Caching**: Engines cached as `{model}_{precision}_batch{max_batch}_{shape}.trt`
- **Metadata**: Engine build info stored alongside cached engines

## Telemetry

- **Sampling Rate**: 200ms intervals during benchmark execution
- **Run Correlation**: Each run gets unique run_id with run-scoped telemetry
- **Relative Timestamps**: `t_ms` measured from run start (t=0)
- **Metrics Collected**: GPU utilization %, memory used/total MB, temperature °C, power usage W
- **Fallback**: Gracefully handles NVML unavailable (returns empty telemetry)

## Supported Models

- **ResNet50**: Deep residual network (25.6M parameters) - ImageNet pretrained

(Extensible - add more models in `runner/app/workloads/`)

## Environment Metadata

Each benchmark run captures:
- GPU name and driver version (via NVML)
- CUDA version
- PyTorch and TensorRT versions
- Python version
- CPU model (macOS: sysctl, Linux: /proc/cpuinfo)
- Git commit hash
- Timestamp
- Sanity check result (pass/fail)

## Run Lifecycle

1. **Client** → POST `/runs` to **Controller** with `RunCreateRequest`
2. **Controller** creates `RunRecord` in SQLite with status=`QUEUED`, returns `run_id`
3. **Background task** updates status to `RUNNING`
4. **Controller** → POST `/execute` to **Runner** for each engine
5. **Runner** executes benchmark, collects telemetry, returns `ExecuteResponse`
6. **Controller** aggregates results, updates status to `SUCCEEDED` or `FAILED`
7. **Client** polls GET `/runs/{run_id}` until status is terminal

## Development

### Running Tests

```bash
# Contracts (run first - foundational)
cd contracts
pip install -e ".[dev]"
pytest -v

# Controller
cd controller
pip install -e ".[dev]"
pytest -v

# Runner (requires GPU)
cd runner
pip install -e ".[dev]"
pytest -v
```

### Local Development (without Docker)

**Terminal 1 - Runner (on GPU host):**
```bash
cd runner
pip install -e .
cd ../contracts && pip install -e . && cd ../runner
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

**Terminal 2 - Controller (on local machine):**
```bash
cd controller
pip install -e .
cd ../contracts && pip install -e . && cd ../controller
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Example Workflow

```bash
# 1. Start runner (GPU host)
curl http://localhost:8001/health
curl http://localhost:8001/version | jq .

# 2. Create run via controller
RUN_ID=$(curl -s -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "runner_url": "http://localhost:8001",
    "model_name": "resnet50",
    "engines": ["pytorch_cpu", "pytorch_cuda", "tensorrt"],
    "batch_sizes": [1, 4, 8],
    "num_iterations": 50
  }' | jq -r '.run_id')

echo "Run ID: $RUN_ID"

# 3. Poll for results
while true; do
  STATUS=$(curl -s http://localhost:8000/runs/$RUN_ID | jq -r '.status')
  echo "Status: $STATUS"
  [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" ]] && break
  sleep 2
done

# 4. Get final results
curl http://localhost:8000/runs/$RUN_ID | jq .
```

## Docker Deployment

### Runner (GPU Host)

```bash
# Build runner image
docker build -t tensorrt-benchlab-runner -f runner/Dockerfile .

# Run with GPU passthrough
docker run -d --name runner \
  --gpus all \
  -p 8001:8001 \
  -v $(pwd)/cache:/app/cache \
  -e BENCHLAB_LOG_LEVEL=INFO \
  tensorrt-benchlab-runner

# Check logs
docker logs -f runner

# Verify GPU access
docker exec runner nvidia-smi
```

### Controller (Local/Cloud)

```bash
# Via docker-compose (TODO)
docker-compose up controller frontend

# Or manually
cd controller
pip install -e .
cd ../contracts && pip install -e .
cd ../controller && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Schema Versioning

All API contracts use semantic versioning via `SCHEMA_VERSION` constant in `contracts/`:
- **Major**: Breaking changes (e.g., 1.0.0 → 2.0.0)
- **Minor**: New fields, backward compatible (e.g., 1.0.0 → 1.1.0)
- **Patch**: Bug fixes (e.g., 1.0.0 → 1.0.1)

Current version: **1.0.0**

All schemas include `schema_version` field for runtime validation.

## Deployment Examples

### AWS EC2 (GPU Instance)

```bash
# Launch g4dn.xlarge or p3.2xlarge instance
# Install Docker + nvidia-container-toolkit
# Clone repo and build runner
docker build -t tensorrt-benchlab-runner -f runner/Dockerfile .
docker run -d --gpus all -p 8001:8001 tensorrt-benchlab-runner

# Point controller at public IP
curl -X POST http://localhost:8000/runs \
  -d '{"runner_url": "http://<ec2-public-ip>:8001", ...}'
```

### RunPod / Lambda Labs

```bash
# SSH into GPU pod
# Clone repo
git clone <repo-url>
cd tensorrt-benchlab

# Build and run
docker build -t tensorrt-benchlab-runner -f runner/Dockerfile .
docker run -d --gpus all -p 8001:8001 tensorrt-benchlab-runner

# Use pod's public IP in controller
```

See [docs/DEPLOY_RUNNER.md](docs/DEPLOY_RUNNER.md) (TODO) for detailed deployment guides.

## Files Reference

**Key Implementation Files:**
- [contracts/contracts/schemas.py](contracts/contracts/schemas.py) - All Pydantic schemas
- [runner/app/engines/tensorrt.py](runner/app/engines/tensorrt.py) - TensorRT implementation
- [runner/app/telemetry/run_telemetry.py](runner/app/telemetry/run_telemetry.py) - Run-scoped telemetry
- [controller/app/main.py](controller/app/main.py) - Controller orchestration
- [controller/app/db/models.py](controller/app/db/models.py) - SQLModel schema
- [runner/Dockerfile](runner/Dockerfile) - NVIDIA CUDA runner container

**Documentation:**
- [PROGRESS.md](PROGRESS.md) - Implementation status and roadmap
- [CLAUDE.md](CLAUDE.md) - AI agent instructions

## License

MIT

## Contributing

This is a weekend project demonstrating production-quality distributed benchmarking infrastructure. Contributions welcome!