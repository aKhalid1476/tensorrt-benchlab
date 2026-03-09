# Telemetry & Metrics Quickstart

## Installation

```bash
cd backend

# Install dependencies (including pynvml and prometheus-client)
pip install -e .

# Or install from requirements
pip install pynvml prometheus-client
```

## Start the Server

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On startup, you should see logs indicating:
```
event=nvml_init device=NVIDIA GeForce RTX 3090 max_samples=1000 interval_ms=200
event=sampling_start interval_ms=200
event=prometheus_update_start
event=startup_complete
```

## Quick Test

```bash
# Test telemetry
bash scripts/test_telemetry.sh

# Or manually:

# 1. Health check
curl http://localhost:8000/health

# 2. Check sampler stats
curl http://localhost:8000/telemetry/stats | jq .

# 3. Get live GPU metrics (wait 1s for samples)
sleep 1
curl http://localhost:8000/telemetry/live | jq .

# 4. Check Prometheus metrics
curl http://localhost:8000/metrics | grep benchlab_gpu

# 5. Start a benchmark to generate metrics
curl -X POST http://localhost:8000/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "engine_type": "pytorch_cpu",
    "batch_sizes": [1],
    "num_iterations": 5,
    "warmup_iterations": 2
  }'

# 6. Check benchmark metrics updated
curl http://localhost:8000/metrics | grep benchlab_benchmark
```

## Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /telemetry/live` | Recent GPU telemetry samples |
| `GET /telemetry/stats` | Sampler status and statistics |
| `GET /metrics` | Prometheus metrics (text format) |

## What's Implemented

### ✅ NVML GPU Telemetry
- Background sampling every 200ms
- Ring buffer with 1000 samples (~200s history)
- Captures: GPU util%, memory, temperature, power
- Auto-starts on app startup
- Graceful shutdown on app stop

### ✅ Prometheus Metrics
- API request latency histograms
- API request counters (by method, endpoint, status)
- Benchmark run duration histograms
- Benchmark iteration latency histograms
- Active benchmark runs gauge
- GPU utilization, memory, temperature, power gauges
- System information (GPU name, CUDA, PyTorch versions)

### ✅ Instrumentation
- Metrics middleware for all API requests
- Benchmark route instrumentation
- Background task to update Prometheus GPU metrics every 1s

## Architecture

```
App Startup
    ├─> Start NVML Sampler (background task, 200ms interval)
    ├─> Start Prometheus Update Loop (background task, 1s interval)
    └─> Set System Info in Prometheus

Background Tasks (Running)
    ├─> NVML Sampler
    │   └─> Sample GPU → Ring Buffer (every 200ms)
    │
    └─> Prometheus Update Loop
        └─> Read latest sample → Update Prometheus gauges (every 1s)

API Request
    ├─> Metrics Middleware (record latency & count)
    └─> Route Handler
        └─> Benchmark Runner (record run duration & iteration latencies)

App Shutdown
    ├─> Stop Prometheus Update Loop
    └─> Stop NVML Sampler (graceful)
```

## Files Created/Modified

### New Files:
- `app/telemetry/prometheus_metrics.py` - Prometheus metrics definitions
- `app/middleware/metrics.py` - Request tracking middleware
- `app/middleware/__init__.py` - Package init
- `scripts/test_telemetry.sh` - Test script
- `TELEMETRY_METRICS.md` - Complete documentation
- `TELEMETRY_QUICKSTART.md` - This file

### Modified Files:
- `app/telemetry/nvml_sampler.py` - Added background sampling
- `app/api/routes_metrics.py` - Added Prometheus update function
- `app/api/routes_bench.py` - Added metrics instrumentation
- `app/main.py` - Added lifespan management, middleware

### Dependencies (Already in pyproject.toml):
- `pynvml>=11.5.0` - NVIDIA GPU monitoring
- `prometheus-client>=0.19.0` - Prometheus metrics

## Next Steps

1. **Install dependencies**: `pip install -e .`
2. **Start server**: `uvicorn app.main:app --reload`
3. **Run tests**: `bash scripts/test_telemetry.sh`
4. **Set up Prometheus** (optional): See [TELEMETRY_METRICS.md](TELEMETRY_METRICS.md#3-prometheus-integration)
5. **Create Grafana dashboard** (optional): See [TELEMETRY_METRICS.md](TELEMETRY_METRICS.md#4-grafana-dashboard)

## Troubleshooting

### "NVML not available" warning
This is normal if:
- No NVIDIA GPU present
- NVIDIA drivers not installed
- Running in Docker without GPU passthrough

The sampler will gracefully fallback and return zero values.

### No samples in /telemetry/live
Wait 1-2 seconds after startup for background sampler to collect samples.

Check sampler status:
```bash
curl http://localhost:8000/telemetry/stats | jq .is_sampling
```

Should return `true`.

## Production Deployment

For production:
1. Set up Prometheus scraping (5-15s interval)
2. Create Grafana dashboards for visualization
3. Configure alerting rules (GPU temp, error rates, etc.)
4. Export logs to centralized logging (ELK, Loki, etc.)
5. Consider longer retention in Prometheus or use TSDB like Victoria Metrics

See [TELEMETRY_METRICS.md](TELEMETRY_METRICS.md) for complete production guide.
