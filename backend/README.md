# TensorRT BenchLab - Backend

FastAPI backend for inference benchmarking.

## API Routes

### Benchmark Routes (`/bench`)

- **POST /bench/run** - Start a new benchmark run
  - Request: `BenchmarkRequest` (model, engine, batch sizes, iterations)
  - Response: `BenchmarkRunResponse` (run_id, status)
  - Returns immediately; benchmark runs in background

- **GET /bench/runs/{run_id}** - Get benchmark result
  - Response: `BenchmarkResult` (full results with statistics)
  - Poll this endpoint to track progress

- **GET /bench/runs** - List recent benchmark runs
  - Query param: `limit` (default: 50)
  - Response: List of `BenchmarkResult`

### Telemetry Routes (`/telemetry`)

- **GET /telemetry/live** - Get recent GPU telemetry
  - Response: `TelemetryResponse` (samples, device_name)
  - Includes GPU utilization, memory, temperature, power

### Metrics

- **GET /metrics** - Prometheus metrics endpoint
  - Standard Prometheus format
  - Python runtime metrics

### Health

- **GET /health** - Health check endpoint

## Pydantic Schemas

All request/response models are defined in `app/schemas/bench.py`:

- `BenchmarkRequest` - Start benchmark request
- `BenchmarkResult` - Complete benchmark result
- `BenchmarkRunResponse` - Run started response
- `BatchStats` - Statistics for single batch size
- `EnvironmentMetadata` - System environment info
- `TelemetrySample` - GPU telemetry sample
- `TelemetryResponse` - Telemetry response with samples

## Structured Logging

All logs use structured format for easy parsing:

```
timestamp=2024-02-13T20:00:00Z level=INFO logger=app.api.routes_bench event=benchmark_start run_id=abc123 model=resnet50
```

Log fields:
- `timestamp` - ISO 8601 UTC timestamp
- `level` - Log level (DEBUG, INFO, WARNING, ERROR)
- `logger` - Logger name
- `event` - Event type
- Custom fields (run_id, model, batch_size, etc.)

## Storage

Results are stored in:
1. In-memory dictionary for active runs
2. JSON files in `./data/` for persistence

Each result is saved as `./data/{run_id}.json` with full schema.

## Running

### Local Development

```bash
pip install -e .[dev]
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t benchlab-backend .
docker run -p 8000:8000 benchlab-backend
```

### Environment Variables

- `BENCHLAB_LOG_LEVEL` - Log level (default: INFO)
- `BENCHLAB_WARMUP_RUNS` - Default warmup runs
- `BENCHLAB_MEASUREMENT_RUNS` - Default measurement runs

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Lint
ruff check app/
mypy app/
```

## Architecture

```
app/
├── main.py              # FastAPI app, lifespan, middleware
├── api/                 # API routes
│   ├── routes_bench.py  # Benchmark endpoints
│   └── routes_metrics.py # Telemetry endpoints
├── bench/               # Benchmark orchestration
│   ├── runner.py        # BenchmarkRunner
│   ├── methodology.py   # Statistics calculations
│   └── workloads.py     # Input generation
├── engines/             # Inference engines
│   ├── base.py          # Abstract engine interface
│   ├── torch_cpu.py     # PyTorch CPU engine
│   ├── torch_cuda.py    # PyTorch CUDA engine
│   └── tensorrt.py      # TensorRT engine (placeholder)
├── schemas/             # Pydantic models
│   └── bench.py         # All API schemas
├── storage/             # Results persistence
│   └── results_store.py # JSON-based store
├── telemetry/           # GPU monitoring
│   └── nvml_sampler.py  # NVML sampling
└── utils/               # Utilities
    ├── env.py           # Environment metadata
    ├── logging_config.py # Structured logging
    └── timing.py        # Timing utilities
```

## Adding New Models

1. Add model loading in `engines/torch_cpu.py` and `torch_cuda.py`:
```python
elif self.model_name == "your_model":
    self.model = models.your_model(weights=...)
```

2. Update frontend model selector in `frontend/src/components/BenchmarkControls.tsx`

## Adding New Engines

1. Create new engine class inheriting from `InferenceEngine`
2. Implement `load_model()` and `infer()` methods
3. Register in `bench/runner.py` engines dict
4. Add to `EngineType` enum in `schemas/bench.py`
