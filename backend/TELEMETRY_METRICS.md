# Telemetry & Metrics Guide

## Overview

TensorRT BenchLab provides comprehensive observability through:
1. **NVML GPU Telemetry**: Real-time GPU metrics sampled every 200ms
2. **Prometheus Metrics**: Production-grade metrics for monitoring and alerting
3. **Structured Logging**: Key-value format logs for easy parsing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  NVML Sampler    │         │  Prometheus      │          │
│  │  (Background)    │────────▶│  Metrics         │          │
│  │  Every 200ms     │         │  Update Loop     │          │
│  └──────────────────┘         └──────────────────┘          │
│         │                              │                     │
│         │                              │                     │
│         ▼                              ▼                     │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Ring Buffer     │         │  Prometheus      │          │
│  │  (1000 samples)  │         │  Gauges/Counters │          │
│  └──────────────────┘         └──────────────────┘          │
│         │                              │                     │
│         ▼                              ▼                     │
│  GET /telemetry/live           GET /metrics                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 1. GPU Telemetry (NVML)

### Background Sampling

The NVML sampler runs continuously in the background:
- **Interval**: 200ms (5 samples/second)
- **Buffer**: Ring buffer with 1000 samples (~200 seconds of history)
- **Auto-start**: Starts on application startup
- **Auto-stop**: Gracefully stops on shutdown

### Metrics Captured

| Metric | Unit | Description |
|--------|------|-------------|
| GPU Utilization | % | GPU compute usage (0-100%) |
| Memory Used | MB | GPU memory currently used |
| Memory Total | MB | Total GPU memory available |
| Temperature | °C | GPU temperature (if available) |
| Power Usage | W | Power consumption (if available) |

### API Endpoints

#### GET /telemetry/live

Returns recent GPU telemetry samples.

**Response:**
```json
{
  "device_name": "NVIDIA GeForce RTX 3090",
  "samples": [
    {
      "timestamp": "2026-02-13T10:30:45.123Z",
      "gpu_utilization_percent": 85.5,
      "memory_used_mb": 12800.0,
      "memory_total_mb": 24576.0,
      "temperature_celsius": 72.0,
      "power_usage_watts": 320.5
    }
  ]
}
```

#### GET /telemetry/stats

Returns sampler statistics.

**Response:**
```json
{
  "enabled": true,
  "device_name": "NVIDIA GeForce RTX 3090",
  "samples_collected": 1000,
  "max_samples": 1000,
  "sample_interval_ms": 200,
  "is_sampling": true
}
```

### Usage Example

```bash
# Get recent telemetry samples
curl http://localhost:8000/telemetry/live | jq .

# Get sampler stats
curl http://localhost:8000/telemetry/stats | jq .
```

## 2. Prometheus Metrics

### Metrics Endpoint

**GET /metrics**

Prometheus-compatible metrics endpoint in text format.

### Available Metrics

#### API Request Metrics

```prometheus
# API request duration histogram
benchlab_api_request_duration_seconds{method="POST", endpoint="/bench/run", status="200"}

# API request counter
benchlab_api_requests_total{method="POST", endpoint="/bench/run", status="200"}
```

**Labels:**
- `method`: HTTP method (GET, POST, etc.)
- `endpoint`: API endpoint path
- `status`: HTTP status code

#### Benchmark Run Metrics

```prometheus
# Benchmark run duration histogram
benchlab_benchmark_run_duration_seconds{model="resnet50", engine="pytorch_cuda", status="completed"}

# Active benchmark runs gauge
benchlab_benchmark_runs_active

# Benchmark runs counter
benchlab_benchmark_runs_total{model="resnet50", engine="pytorch_cuda", status="completed"}

# Iteration latency histogram
benchlab_benchmark_iteration_latency_ms{model="resnet50", engine="pytorch_cuda", batch_size="4"}
```

**Labels:**
- `model`: Model name (resnet50, mobilenet_v2)
- `engine`: Engine type (pytorch_cpu, pytorch_cuda, tensorrt)
- `batch_size`: Batch size used
- `status`: Run status (completed, failed)

#### GPU Metrics

```prometheus
# GPU utilization percentage
benchlab_gpu_utilization_percent{device="NVIDIA GeForce RTX 3090"}

# GPU memory used (MB)
benchlab_gpu_memory_used_mb{device="NVIDIA GeForce RTX 3090"}

# GPU memory total (MB)
benchlab_gpu_memory_total_mb{device="NVIDIA GeForce RTX 3090"}

# GPU temperature (Celsius)
benchlab_gpu_temperature_celsius{device="NVIDIA GeForce RTX 3090"}

# GPU power usage (Watts)
benchlab_gpu_power_usage_watts{device="NVIDIA GeForce RTX 3090"}
```

**Labels:**
- `device`: GPU device name

#### System Information

```prometheus
# System information
benchlab_system_info{gpu_name="NVIDIA GeForce RTX 3090", cuda_version="12.1", torch_version="2.1.0"}
```

### Usage Examples

```bash
# Get all metrics
curl http://localhost:8000/metrics

# Get GPU metrics only
curl http://localhost:8000/metrics | grep benchlab_gpu

# Get benchmark metrics only
curl http://localhost:8000/metrics | grep benchlab_benchmark
```

## 3. Prometheus Integration

### Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'tensorrt-benchlab'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Sample Queries

#### GPU Utilization Over Time
```promql
benchlab_gpu_utilization_percent{device="NVIDIA GeForce RTX 3090"}
```

#### Average API Latency (p50)
```promql
histogram_quantile(0.50,
  rate(benchlab_api_request_duration_seconds_bucket[5m])
)
```

#### API Error Rate
```promql
rate(benchlab_api_requests_total{status=~"5.."}[5m])
```

#### Benchmark Run Success Rate
```promql
rate(benchlab_benchmark_runs_total{status="completed"}[1h])
/
rate(benchlab_benchmark_runs_total[1h])
```

#### GPU Memory Utilization %
```promql
(benchlab_gpu_memory_used_mb / benchlab_gpu_memory_total_mb) * 100
```

## 4. Grafana Dashboard

### Recommended Panels

1. **GPU Utilization**
   - Type: Time series
   - Query: `benchlab_gpu_utilization_percent`

2. **GPU Memory Usage**
   - Type: Time series
   - Query: `benchlab_gpu_memory_used_mb`, `benchlab_gpu_memory_total_mb`

3. **API Request Latency (p50, p95)**
   - Type: Time series
   - Queries:
     ```promql
     histogram_quantile(0.50, rate(benchlab_api_request_duration_seconds_bucket[5m]))
     histogram_quantile(0.95, rate(benchlab_api_request_duration_seconds_bucket[5m]))
     ```

4. **Active Benchmark Runs**
   - Type: Gauge
   - Query: `benchlab_benchmark_runs_active`

5. **Benchmark Run Duration by Engine**
   - Type: Heatmap
   - Query: `benchlab_benchmark_run_duration_seconds`

## 5. Alerting Rules

### Sample Alerts

```yaml
groups:
  - name: tensorrt_benchlab
    interval: 30s
    rules:
      # GPU temperature too high
      - alert: GPUTemperatureHigh
        expr: benchlab_gpu_temperature_celsius > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU temperature is high"
          description: "GPU {{ $labels.device }} temperature is {{ $value }}°C"

      # GPU utilization low during benchmark
      - alert: LowGPUUtilization
        expr: |
          benchlab_benchmark_runs_active > 0
          and
          benchlab_gpu_utilization_percent < 20
        for: 2m
        labels:
          severity: info
        annotations:
          summary: "Low GPU utilization during benchmark"

      # API error rate high
      - alert: HighAPIErrorRate
        expr: |
          rate(benchlab_api_requests_total{status=~"5.."}[5m])
          /
          rate(benchlab_api_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Benchmark run taking too long
      - alert: BenchmarkRunSlow
        expr: |
          benchlab_benchmark_run_duration_seconds > 300
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Benchmark run is taking too long"
          description: "Run {{ $labels.model }}/{{ $labels.engine }} took {{ $value }}s"
```

## 6. Implementation Details

### NVML Sampler

**File**: `app/telemetry/nvml_sampler.py`

- Uses `pynvml` library for NVIDIA GPU access
- Runs in asyncio background task
- Automatic fallback when NVML unavailable
- Graceful error handling

### Prometheus Metrics

**File**: `app/telemetry/prometheus_metrics.py`

- Uses `prometheus_client` library
- Defines all metric types (Counter, Gauge, Histogram, Info)
- Helper functions for recording metrics

### Metrics Middleware

**File**: `app/middleware/metrics.py`

- Automatically tracks all API requests
- Records latency and status codes
- Skips `/metrics` endpoint to avoid recursion

### Lifecycle Management

**File**: `app/main.py`

The lifespan context manager handles:
1. Starting NVML background sampler on startup
2. Starting Prometheus metrics update loop
3. Graceful shutdown of both tasks
4. Setting system info in Prometheus

## 7. Testing

### Manual Testing

```bash
# Run test script
cd backend
bash scripts/test_telemetry.sh
```

### Verify Sampling

```bash
# Start server
uvicorn app.main:app --reload

# In another terminal, check stats
curl http://localhost:8000/telemetry/stats | jq .

# Wait a few seconds for samples
sleep 5

# Get live telemetry
curl http://localhost:8000/telemetry/live | jq '.samples | length'
```

### Verify Prometheus Metrics

```bash
# Get metrics
curl http://localhost:8000/metrics

# Trigger a benchmark
curl -X POST http://localhost:8000/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "engine_type": "pytorch_cpu",
    "batch_sizes": [1],
    "num_iterations": 5,
    "warmup_iterations": 2
  }'

# Check metrics updated
curl http://localhost:8000/metrics | grep benchlab_benchmark
```

## 8. Troubleshooting

### NVML Not Available

If NVML initialization fails:
- Check NVIDIA drivers installed: `nvidia-smi`
- Install pynvml: `pip install pynvml`
- Sampler will fallback gracefully with warning

### No Samples Collected

Check sampler status:
```bash
curl http://localhost:8000/telemetry/stats | jq .
```

If `is_sampling: false`:
- Check logs for errors
- Verify CUDA is available
- Restart application

### Prometheus Metrics Not Updating

1. Check `/metrics` endpoint accessible:
   ```bash
   curl http://localhost:8000/metrics | head
   ```

2. Check GPU metrics update loop running:
   ```bash
   # Look for logs
   docker logs <container> | grep prometheus_update
   ```

3. Verify telemetry has samples:
   ```bash
   curl http://localhost:8000/telemetry/stats
   ```

## 9. Performance Impact

### NVML Sampling
- **CPU**: < 0.1% (samples are lightweight)
- **Memory**: ~100KB for 1000 samples
- **Latency**: No impact on API requests (runs in background)

### Prometheus Metrics
- **CPU**: < 0.1% (metrics are in-memory counters)
- **Memory**: ~1MB for all metrics
- **Latency**: < 1ms added to each request (middleware)

### Recommendations
- Default settings are production-ready
- Adjust `sample_interval_ms` if needed (200ms is recommended)
- Adjust `max_samples` for longer history (default 1000 = ~200s)

## 10. Best Practices

1. **Monitor GPU metrics during benchmarks** to ensure full utilization
2. **Set up alerts** for high temperature or low utilization
3. **Use Prometheus recording rules** for complex queries
4. **Archive metrics** to long-term storage (e.g., Victoria Metrics)
5. **Create Grafana dashboards** for visualization
6. **Export metrics** to your observability platform (Datadog, New Relic, etc.)
