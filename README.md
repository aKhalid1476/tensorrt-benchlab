# ⚡ TensorRT BenchLab

Production-quality benchmarking platform for comparing inference performance across PyTorch CPU, PyTorch GPU, and TensorRT engines.

## Features

- **Multiple Inference Engines**: Benchmark PyTorch CPU, PyTorch GPU, and TensorRT
- **Reproducible Methodology**: Fixed random seeds, configurable warmup/iterations, stable statistics (p50/p95/mean/stddev/throughput)
- **Live Dashboard**: Real-time GPU telemetry and benchmark results visualization
- **Production Observability**: Prometheus metrics, environment metadata tracking
- **Docker-Ready**: Complete docker-compose setup for easy deployment

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA drivers (for GPU benchmarks)
- Python 3.11+ (for local development)
- Node 20+ (for frontend development)

### Run with Docker

```bash
# Start all services
make dev

# Or manually
docker-compose up
```

Access:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Prometheus Metrics: http://localhost:8000/metrics

### Local Development

#### Backend

```bash
cd backend
pip install -e .[dev]
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
tensorrt-benchlab/
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── bench/        # Benchmark orchestration
│   │   ├── engines/      # Inference engine implementations
│   │   ├── schemas/      # Pydantic schemas
│   │   ├── storage/      # Results persistence
│   │   ├── telemetry/    # GPU metrics collection
│   │   └── utils/        # Environment metadata, timing
│   ├── tests/            # Backend tests
│   └── pyproject.toml
├── frontend/             # React + TypeScript frontend
│   ├── src/
│   │   ├── api/          # API client
│   │   ├── components/   # React components
│   │   └── pages/        # Page components
│   └── package.json
├── docker-compose.yml    # Container orchestration
├── Makefile             # Development commands
└── CLAUDE.md            # AI agent instructions

```

## Available Commands

```bash
make help           # Show all available commands
make dev            # Run development servers
make build          # Build Docker images
make test           # Run tests
make lint           # Run linters
make bench          # Quick benchmark test
make clean          # Clean generated files
```

## API Endpoints

### Benchmark

- `POST /bench/run` - Start a benchmark run
- `GET /bench/runs/{run_id}` - Get benchmark result
- `GET /bench/runs` - List recent runs

### Telemetry

- `GET /telemetry/live` - Get recent GPU telemetry samples

### Metrics

- `GET /metrics` - Prometheus metrics endpoint

## Benchmark Methodology

- **Warmup**: Configurable warmup iterations (default: 3) excluded from measurements
- **Measurement**: N iterations (default: 100) with precise timing
- **Batch Sizes**: [1, 4, 8, 16] by default, configurable
- **Statistics**: p50, p95, mean, stddev latency + throughput (req/s)
- **Reproducibility**: Fixed random seed (42), fixed input tensors
- **Metadata**: Captures GPU, CUDA, PyTorch versions, git commit, timestamp

## Supported Models

- ResNet50
- MobileNetV2

(Extensible - add more in `backend/app/engines/`)

## Environment Metadata

Each benchmark run captures:
- GPU name and driver version
- CUDA version
- PyTorch and TensorRT versions
- CPU model
- Git commit hash
- Timestamp

## API Testing

### Quick Test Script

```bash
./backend/scripts/test_api.sh
```

### Manual curl Examples

See [CURL_EXAMPLES.md](CURL_EXAMPLES.md) for comprehensive API testing examples including:
- Starting benchmarks with different configurations
- Polling for results
- GPU telemetry
- Error handling
- Complete workflow scripts

### Quick Examples

Start a benchmark:
```bash
curl -X POST http://localhost:8000/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "engine_type": "pytorch_cpu",
    "batch_sizes": [1, 4, 8, 16],
    "num_iterations": 100,
    "warmup_iterations": 3
  }'
```

Get results (replace `{run_id}` with the ID from above):
```bash
curl http://localhost:8000/bench/runs/{run_id} | jq '.'
```

## Development

### Running Tests

```bash
# Backend
cd backend && pytest tests/ -v

# Frontend
cd frontend && npm run test
```

### Linting

```bash
# Backend (ruff + mypy)
cd backend && ruff check app/ && mypy app/

# Frontend (ESLint)
cd frontend && npm run lint
```

## License

MIT

## Contributing

This is a weekend project demonstrating production-quality benchmarking infrastructure. Contributions welcome!