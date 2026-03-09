#!/bin/bash
# Test telemetry and metrics endpoints

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "========================================="
echo "Testing Telemetry & Metrics Endpoints"
echo "========================================="
echo ""

# Health check
echo "1. Health Check"
echo "   GET $BASE_URL/health"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""
echo ""

# Telemetry stats
echo "2. Telemetry Sampler Stats"
echo "   GET $BASE_URL/telemetry/stats"
curl -s "$BASE_URL/telemetry/stats" | python3 -m json.tool
echo ""
echo ""

# Live telemetry (wait a bit for samples)
echo "3. Live GPU Telemetry (waiting 1s for samples)"
sleep 1
echo "   GET $BASE_URL/telemetry/live"
response=$(curl -s "$BASE_URL/telemetry/live")
echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Device: {data[\"device_name\"]}')
print(f'Samples: {len(data[\"samples\"])}')
if data['samples']:
    last = data['samples'][-1]
    print(f'Latest Sample:')
    print(f'  - GPU Util: {last[\"gpu_utilization_percent\"]:.1f}%')
    print(f'  - Memory: {last[\"memory_used_mb\"]:.1f} / {last[\"memory_total_mb\"]:.1f} MB')
    if last['temperature_celsius']:
        print(f'  - Temperature: {last[\"temperature_celsius\"]:.1f}°C')
    if last['power_usage_watts']:
        print(f'  - Power: {last[\"power_usage_watts\"]:.1f}W')
"
echo ""
echo ""

# Prometheus metrics
echo "4. Prometheus Metrics"
echo "   GET $BASE_URL/metrics"
echo "   (Showing GPU-related metrics only)"
curl -s "$BASE_URL/metrics" | grep -E "^benchlab_(gpu|system)" | head -20
echo ""
echo ""

# Start a small benchmark run to test metrics
echo "5. Starting benchmark run to test metrics"
run_response=$(curl -s -X POST "$BASE_URL/bench/run" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "engine_type": "pytorch_cpu",
    "batch_sizes": [1],
    "num_iterations": 5,
    "warmup_iterations": 2
  }')

run_id=$(echo "$run_response" | python3 -c "import sys, json; print(json.load(sys.stdin)['run_id'])")
echo "   Run ID: $run_id"
echo ""

# Wait for completion
echo "6. Waiting for benchmark completion..."
max_wait=30
elapsed=0
while [ $elapsed -lt $max_wait ]; do
    status=$(curl -s "$BASE_URL/bench/runs/$run_id" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
    echo "   Status: $status (${elapsed}s elapsed)"

    if [ "$status" = "completed" ] || [ "$status" = "failed" ]; then
        break
    fi

    sleep 2
    elapsed=$((elapsed + 2))
done
echo ""

# Check Prometheus metrics after benchmark
echo "7. Prometheus Benchmark Metrics"
echo "   (Showing benchmark-related metrics)"
curl -s "$BASE_URL/metrics" | grep -E "^benchlab_benchmark" | head -20
echo ""
echo ""

# Check API request metrics
echo "8. Prometheus API Request Metrics"
curl -s "$BASE_URL/metrics" | grep -E "^benchlab_api" | head -20
echo ""
echo ""

echo "========================================="
echo "Telemetry & Metrics Tests Complete!"
echo "========================================="
