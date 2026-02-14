# API Testing with curl

Complete examples for testing the TensorRT BenchLab API.

## Prerequisites

Ensure the backend is running:
```bash
# Using Docker
make dev

# Or locally
cd backend && uvicorn app.main:app --reload
```

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "tensorrt-benchlab"
}
```

## Start a Benchmark Run

### Example 1: Quick CPU Benchmark (ResNet50, single batch)

```bash
curl -X POST http://localhost:8000/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "engine_type": "pytorch_cpu",
    "batch_sizes": [1],
    "num_iterations": 10,
    "warmup_iterations": 2
  }'
```

Expected response:
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Benchmark run 550e8400-e29b-41d4-a716-446655440000 started"
}
```

**Save the run_id for the next step!**

### Example 2: Full Benchmark (All batch sizes, 100 iterations)

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

### Example 3: GPU Benchmark

```bash
curl -X POST http://localhost:8000/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "engine_type": "pytorch_gpu",
    "batch_sizes": [1, 4, 8, 16],
    "num_iterations": 100,
    "warmup_iterations": 3
  }'
```

### Example 4: MobileNetV2 Benchmark

```bash
curl -X POST http://localhost:8000/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mobilenet_v2",
    "engine_type": "pytorch_cpu",
    "batch_sizes": [1, 4, 8, 16],
    "num_iterations": 100,
    "warmup_iterations": 3
  }'
```

## Get Benchmark Results

Replace `{run_id}` with the ID from the POST response:

```bash
# Poll for results
curl http://localhost:8000/bench/runs/550e8400-e29b-41d4-a716-446655440000
```

Expected response (pending):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_name": "resnet50",
  "engine_type": "pytorch_cpu",
  "status": "pending",
  "results": [],
  "created_at": "2024-02-13T20:00:00",
  "completed_at": null,
  ...
}
```

Expected response (completed):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_name": "resnet50",
  "engine_type": "pytorch_cpu",
  "status": "completed",
  "results": [
    {
      "batch_size": 1,
      "latency_p50_ms": 45.2,
      "latency_p95_ms": 48.1,
      "latency_mean_ms": 45.5,
      "latency_stddev_ms": 1.2,
      "throughput_req_per_sec": 22.0
    },
    {
      "batch_size": 4,
      "latency_p50_ms": 150.3,
      "latency_p95_ms": 155.2,
      "latency_mean_ms": 151.0,
      "latency_stddev_ms": 2.1,
      "throughput_req_per_sec": 26.5
    }
  ],
  "environment": {
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "cuda_version": "12.1",
    "torch_version": "2.1.0",
    "cpu_model": "Intel Core i9-12900K",
    "timestamp": "2024-02-13T20:00:05",
    "git_commit": "abc123def456"
  },
  "created_at": "2024-02-13T20:00:00",
  "completed_at": "2024-02-13T20:02:30"
}
```

## List Recent Benchmark Runs

```bash
# Get last 10 runs
curl http://localhost:8000/bench/runs?limit=10
```

Expected response:
```json
[
  {
    "run_id": "...",
    "model_name": "resnet50",
    "status": "completed",
    ...
  },
  {
    "run_id": "...",
    "model_name": "mobilenet_v2",
    "status": "running",
    ...
  }
]
```

## Get GPU Telemetry

```bash
curl http://localhost:8000/telemetry/live
```

Expected response:
```json
{
  "device_name": "NVIDIA GeForce RTX 3090",
  "samples": [
    {
      "timestamp": "2024-02-13T20:00:00",
      "gpu_utilization_percent": 85.5,
      "memory_used_mb": 12000.0,
      "memory_total_mb": 24000.0,
      "temperature_celsius": 72.0,
      "power_usage_watts": 320.5
    },
    ...
  ]
}
```

## Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Expected response (Prometheus format):
```
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 1234.0
...
```

## Interactive API Documentation

Visit these URLs in your browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Complete Workflow Example

```bash
#!/bin/bash
# complete_benchmark.sh - Run a complete benchmark workflow

echo "==> Starting benchmark..."
RESPONSE=$(curl -s -X POST http://localhost:8000/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "engine_type": "pytorch_cpu",
    "batch_sizes": [1, 4, 8],
    "num_iterations": 50,
    "warmup_iterations": 3
  }')

RUN_ID=$(echo $RESPONSE | jq -r '.run_id')
echo "Benchmark started with run_id: $RUN_ID"

echo "==> Polling for results..."
while true; do
  STATUS=$(curl -s http://localhost:8000/bench/runs/$RUN_ID | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi

  sleep 2
done

echo "==> Fetching final results..."
curl -s http://localhost:8000/bench/runs/$RUN_ID | jq '.'

echo "==> Done!"
```

Make it executable and run:
```bash
chmod +x complete_benchmark.sh
./complete_benchmark.sh
```

## Error Handling

### Invalid Model Name

```bash
curl -X POST http://localhost:8000/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "invalid_model",
    "engine_type": "pytorch_cpu",
    "batch_sizes": [1]
  }'
```

The benchmark will start but fail during execution. Check status:
```bash
curl http://localhost:8000/bench/runs/{run_id}
```

Response:
```json
{
  "status": "failed",
  "error_message": "Unsupported model: invalid_model",
  ...
}
```

### Invalid Engine Type

```bash
curl -X POST http://localhost:8000/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "engine_type": "invalid_engine",
    "batch_sizes": [1]
  }'
```

Response (422 Validation Error):
```json
{
  "detail": [
    {
      "loc": ["body", "engine_type"],
      "msg": "value is not a valid enumeration member",
      "type": "type_error.enum"
    }
  ]
}
```

## Tips

1. **Use jq for pretty output**:
   ```bash
   curl http://localhost:8000/bench/runs/{run_id} | jq '.'
   ```

2. **Save results to file**:
   ```bash
   curl http://localhost:8000/bench/runs/{run_id} > results.json
   ```

3. **Extract specific fields**:
   ```bash
   # Get only p50 latencies
   curl http://localhost:8000/bench/runs/{run_id} | jq '.results[].latency_p50_ms'
   ```

4. **Monitor in real-time**:
   ```bash
   watch -n 2 "curl -s http://localhost:8000/bench/runs/{run_id} | jq '.status'"
   ```
