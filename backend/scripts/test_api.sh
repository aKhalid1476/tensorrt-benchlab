#!/bin/bash
# Quick API testing script

set -e

API_BASE="http://localhost:8000"

echo "==> Testing TensorRT BenchLab API"
echo ""

# Health check
echo "1. Health Check"
curl -s ${API_BASE}/health | jq '.'
echo ""

# Start benchmark
echo "2. Starting Quick Benchmark (ResNet50, CPU, batch_size=1)"
RESPONSE=$(curl -s -X POST ${API_BASE}/bench/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "engine_type": "pytorch_cpu",
    "batch_sizes": [1],
    "num_iterations": 5,
    "warmup_iterations": 1
  }')

echo "$RESPONSE" | jq '.'
RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id')
echo ""
echo "Run ID: $RUN_ID"
echo ""

# Poll for results
echo "3. Polling for Results (will take ~10-30 seconds)..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo -n "."

  STATUS=$(curl -s ${API_BASE}/bench/runs/${RUN_ID} | jq -r '.status')

  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    echo ""
    break
  fi

  sleep 2
done

echo ""
echo "4. Final Results:"
curl -s ${API_BASE}/bench/runs/${RUN_ID} | jq '.'
echo ""

# Telemetry
echo "5. GPU Telemetry (last 5 samples):"
curl -s ${API_BASE}/telemetry/live | jq '.samples[-5:]'
echo ""

echo "==> Test Complete!"
echo "Visit http://localhost:8000/docs for interactive API documentation"
