#!/bin/bash
# TensorRT BenchLab - End-to-End Demo Script
#
# This script demonstrates the complete workflow:
# 1. Verify runner and controller are healthy
# 2. Create a benchmark run
# 3. Poll until completion
# 4. Display results summary

set -e  # Exit on error

# Configuration
CONTROLLER_URL="${CONTROLLER_URL:-http://localhost:8000}"
RUNNER_URL="${RUNNER_URL:-http://localhost:8001}"
MODEL_NAME="${MODEL_NAME:-resnet50}"
ENGINES="${ENGINES:-pytorch_cpu,pytorch_cuda,tensorrt}"
BATCH_SIZES="${BATCH_SIZES:-1,4,8,16}"
NUM_ITERATIONS="${NUM_ITERATIONS:-50}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-10}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "⚡ TensorRT BenchLab - Demo Script"
echo "=================================="
echo ""

# ==============================================================================
# Step 1: Health Checks
# ==============================================================================

echo -e "${BLUE}Step 1: Checking service health...${NC}"
echo ""

echo -n "  Checking runner ($RUNNER_URL)... "
if curl -sf "$RUNNER_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
    echo ""
    echo "Error: Runner is not healthy. Please start the runner first:"
    echo "  make run-runner"
    echo ""
    exit 1
fi

echo -n "  Checking controller ($CONTROLLER_URL)... "
if curl -sf "$CONTROLLER_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
    echo ""
    echo "Error: Controller is not healthy. Please start the controller first:"
    echo "  make dev-controller"
    echo ""
    exit 1
fi

echo ""

# ==============================================================================
# Step 2: Get Runner Info
# ==============================================================================

echo -e "${BLUE}Step 2: Fetching runner environment info...${NC}"
echo ""

RUNNER_INFO=$(curl -s "$RUNNER_URL/version")
GPU_NAME=$(echo "$RUNNER_INFO" | jq -r '.gpu_name // "N/A"')
CUDA_VERSION=$(echo "$RUNNER_INFO" | jq -r '.cuda_version // "N/A"')
TORCH_VERSION=$(echo "$RUNNER_INFO" | jq -r '.torch_version // "N/A"')
TENSORRT_VERSION=$(echo "$RUNNER_INFO" | jq -r '.tensorrt_version // "N/A"')

echo "  GPU: $GPU_NAME"
echo "  CUDA: $CUDA_VERSION"
echo "  PyTorch: $TORCH_VERSION"
echo "  TensorRT: $TENSORRT_VERSION"
echo ""

# ==============================================================================
# Step 3: Create Benchmark Run
# ==============================================================================

echo -e "${BLUE}Step 3: Creating benchmark run...${NC}"
echo ""

# Convert comma-separated strings to JSON arrays
ENGINES_JSON=$(echo $ENGINES | sed 's/,/","/g' | sed 's/^/"/' | sed 's/$/"/')
BATCH_SIZES_JSON=$(echo $BATCH_SIZES | sed 's/,/, /g')

REQUEST_PAYLOAD=$(cat <<EOF
{
  "runner_url": "$RUNNER_URL",
  "model_name": "$MODEL_NAME",
  "engines": [$ENGINES_JSON],
  "batch_sizes": [$BATCH_SIZES_JSON],
  "num_iterations": $NUM_ITERATIONS,
  "warmup_iterations": $WARMUP_ITERATIONS
}
EOF
)

echo "  Configuration:"
echo "    Model: $MODEL_NAME"
echo "    Engines: $ENGINES"
echo "    Batch sizes: $BATCH_SIZES"
echo "    Iterations: $NUM_ITERATIONS (+ $WARMUP_ITERATIONS warmup)"
echo ""

RESPONSE=$(curl -s -X POST "$CONTROLLER_URL/runs" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_PAYLOAD")

RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id')
STATUS=$(echo "$RESPONSE" | jq -r '.status')

if [ "$RUN_ID" == "null" ] || [ -z "$RUN_ID" ]; then
    echo -e "${RED}Error: Failed to create run${NC}"
    echo "$RESPONSE" | jq .
    exit 1
fi

echo -e "  ${GREEN}✓ Run created${NC}"
echo "  Run ID: $RUN_ID"
echo "  Initial status: $STATUS"
echo ""

# ==============================================================================
# Step 4: Poll for Completion
# ==============================================================================

echo -e "${BLUE}Step 4: Waiting for completion...${NC}"
echo ""

START_TIME=$(date +%s)
POLL_INTERVAL=2  # seconds
MAX_WAIT=300     # 5 minutes

while true; do
    # Get current status
    RUN_DATA=$(curl -s "$CONTROLLER_URL/runs/$RUN_ID")
    CURRENT_STATUS=$(echo "$RUN_DATA" | jq -r '.status')

    # Calculate elapsed time
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))

    echo -ne "  Status: $CURRENT_STATUS (${ELAPSED}s elapsed)\r"

    # Check if terminal status
    if [ "$CURRENT_STATUS" == "succeeded" ]; then
        echo -e "\n  ${GREEN}✓ Benchmark completed successfully!${NC}"
        echo ""
        break
    elif [ "$CURRENT_STATUS" == "failed" ]; then
        echo -e "\n  ${RED}✗ Benchmark failed${NC}"
        echo ""
        ERROR_MSG=$(echo "$RUN_DATA" | jq -r '.error_message // "Unknown error"')
        echo "  Error: $ERROR_MSG"
        echo ""
        exit 1
    elif [ "$CURRENT_STATUS" == "cancelled" ]; then
        echo -e "\n  ${YELLOW}⚠ Benchmark was cancelled${NC}"
        echo ""
        exit 1
    fi

    # Check timeout
    if [ $ELAPSED -gt $MAX_WAIT ]; then
        echo -e "\n  ${RED}✗ Timeout: Benchmark did not complete within ${MAX_WAIT}s${NC}"
        echo ""
        exit 1
    fi

    sleep $POLL_INTERVAL
done

# ==============================================================================
# Step 5: Display Results Summary
# ==============================================================================

echo -e "${BLUE}Step 5: Results Summary${NC}"
echo ""

# Get final results
FINAL_DATA=$(curl -s "$CONTROLLER_URL/runs/$RUN_ID")

# Extract results array
RESULTS=$(echo "$FINAL_DATA" | jq -r '.results')
NUM_RESULTS=$(echo "$RESULTS" | jq 'length')

if [ "$NUM_RESULTS" -eq 0 ]; then
    echo "  No results available."
    echo ""
    exit 0
fi

# Print header
printf "  %-15s %-10s %10s %10s %10s %15s\n" \
    "Engine" "Batch" "p50 (ms)" "p95 (ms)" "Mean (ms)" "Throughput"
echo "  -------------------------------------------------------------------------------"

# Print each result
echo "$RESULTS" | jq -r '.[] |
    "\(.engine_type) \(.batch_size) \(.latency_ms_p50) \(.latency_ms_p95) \(.latency_ms_mean) \(.throughput_img_per_sec)"' | \
while read -r engine batch p50 p95 mean throughput; do
    printf "  %-15s %-10s %10.2f %10.2f %10.2f %12.1f img/s\n" \
        "$engine" "$batch" "$p50" "$p95" "$mean" "$throughput"
done

echo ""

# Display timing breakdown
TOTAL_DURATION=$(echo "$FINAL_DATA" | jq -r '.timing.total_duration_sec // 0')
MODEL_LOAD=$(echo "$FINAL_DATA" | jq -r '.timing.model_load_sec // 0')
WARMUP=$(echo "$FINAL_DATA" | jq -r '.timing.warmup_duration_sec // 0')
MEASUREMENT=$(echo "$FINAL_DATA" | jq -r '.timing.measurement_duration_sec // 0')

echo "  Timing Breakdown:"
printf "    Total duration:       %6.2f s\n" "$TOTAL_DURATION"
printf "    Model loading:        %6.2f s\n" "$MODEL_LOAD"
printf "    Warmup:               %6.2f s\n" "$WARMUP"
printf "    Measurement:          %6.2f s\n" "$MEASUREMENT"
echo ""

# Display telemetry info
TELEMETRY_SAMPLES=$(echo "$FINAL_DATA" | jq -r '.telemetry.samples | length')
echo "  Telemetry: $TELEMETRY_SAMPLES samples collected"
echo ""

# Display environment info
SANITY_CHECK=$(echo "$FINAL_DATA" | jq -r '.environment.sanity_check_passed // "unknown"')
echo "  Sanity Check: $SANITY_CHECK"
echo ""

# ==============================================================================
# Finish
# ==============================================================================

echo -e "${GREEN}✓ Demo complete!${NC}"
echo ""
echo "View full results:"
echo "  curl http://localhost:8000/runs/$RUN_ID | jq ."
echo ""
echo "Or in browser:"
echo "  http://localhost:8000/runs/$RUN_ID"
echo ""
