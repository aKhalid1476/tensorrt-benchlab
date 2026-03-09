# Prometheus Metrics Reference

This document describes all Prometheus metrics exposed by TensorRT BenchLab services.

## Table of Contents
- [Controller Metrics](#controller-metrics)
- [Runner Metrics](#runner-metrics)
- [Metric Types](#metric-types)
- [Usage Examples](#usage-examples)

---

## Controller Metrics

**Endpoint:** `http://localhost:8000/metrics`

### Run Lifecycle Metrics

#### `benchlab_controller_runs_total`
- **Type:** Counter
- **Description:** Total number of benchmark runs created
- **Labels:**
  - `model`: Model name (e.g., "resnet50")
  - `status`: Final run status ("succeeded", "failed")
- **Example:**
  ```
  benchlab_controller_runs_total{model="resnet50",status="succeeded"} 42
  benchlab_controller_runs_total{model="resnet50",status="failed"} 3
  ```

#### `benchlab_controller_runs_by_status`
- **Type:** Gauge
- **Description:** Current number of runs in each status
- **Labels:**
  - `status`: Run status ("queued", "running", "succeeded", "failed", "cancelled")
- **Example:**
  ```
  benchlab_controller_runs_by_status{status="queued"} 2
  benchlab_controller_runs_by_status{status="running"} 1
  benchlab_controller_runs_by_status{status="succeeded"} 45
  ```

#### `benchlab_controller_run_duration_seconds`
- **Type:** Histogram
- **Description:** Duration of benchmark runs from creation to completion
- **Labels:**
  - `model`: Model name
  - `status`: Final run status
- **Buckets:** [1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
- **Example:**
  ```
  benchlab_controller_run_duration_seconds_sum{model="resnet50",status="succeeded"} 450.5
  benchlab_controller_run_duration_seconds_count{model="resnet50",status="succeeded"} 10
  benchlab_controller_run_duration_seconds_bucket{model="resnet50",status="succeeded",le="60"} 7
  ```

#### `benchlab_controller_active_runs`
- **Type:** Gauge
- **Description:** Number of currently active (running) benchmark runs
- **Example:**
  ```
  benchlab_controller_active_runs 2
  ```

### Runner Communication Metrics

#### `benchlab_controller_runner_request_duration_seconds`
- **Type:** Histogram
- **Description:** Duration of runner API requests (POST /execute calls)
- **Labels:**
  - `engine`: Engine type ("pytorch_cpu", "pytorch_cuda", "tensorrt")
  - `status`: Request outcome ("success", "failure")
- **Buckets:** [0.1, 0.5, 1, 5, 10, 30, 60, 120, 300]
- **Example:**
  ```
  benchlab_controller_runner_request_duration_seconds_sum{engine="tensorrt",status="success"} 125.3
  benchlab_controller_runner_request_duration_seconds_count{engine="tensorrt",status="success"} 15
  ```

#### `benchlab_controller_runner_request_retries_total`
- **Type:** Counter
- **Description:** Total number of runner request retries (exponential backoff)
- **Labels:**
  - `engine`: Engine type
  - `attempt`: Retry attempt number ("2", "3", etc.)
- **Example:**
  ```
  benchlab_controller_runner_request_retries_total{engine="pytorch_cuda",attempt="2"} 5
  benchlab_controller_runner_request_retries_total{engine="tensorrt",attempt="3"} 1
  ```

### Controller Info

#### `benchlab_controller_info`
- **Type:** Info
- **Description:** Controller version and configuration information
- **Labels:** `version`, `service`, `python_version`
- **Example:**
  ```
  benchlab_controller_info{version="0.1.0",service="tensorrt-benchlab-controller",python_version="3.11"} 1
  ```

---

## Runner Metrics

**Endpoint:** `http://localhost:8001/metrics`

### Benchmark Execution Metrics

#### `benchlab_runner_benchmark_executions_total`
- **Type:** Counter
- **Description:** Total number of benchmark executions
- **Labels:**
  - `model`: Model name
  - `engine`: Engine type
  - `status`: Execution outcome ("succeeded", "failed")
- **Example:**
  ```
  benchlab_runner_benchmark_executions_total{model="resnet50",engine="tensorrt",status="succeeded"} 50
  benchlab_runner_benchmark_executions_total{model="resnet50",engine="pytorch_cuda",status="failed"} 2
  ```

#### `benchlab_runner_benchmark_duration_seconds`
- **Type:** Histogram
- **Description:** Total duration of benchmark execution (including warmup)
- **Labels:**
  - `model`: Model name
  - `engine`: Engine type
- **Buckets:** [1, 5, 10, 30, 60, 120, 300, 600]
- **Example:**
  ```
  benchlab_runner_benchmark_duration_seconds_sum{model="resnet50",engine="tensorrt"} 450.2
  benchlab_runner_benchmark_duration_seconds_count{model="resnet50",engine="tensorrt"} 25
  ```

### Inference Performance Metrics

#### `benchlab_runner_inference_latency_seconds`
- **Type:** Histogram
- **Description:** Inference latency per batch (forward pass only)
- **Labels:**
  - `model`: Model name
  - `engine`: Engine type
  - `batch_size`: Batch size
- **Buckets:** [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
- **Use Case:** Track actual inference performance across different batch sizes
- **Example:**
  ```
  benchlab_runner_inference_latency_seconds_bucket{model="resnet50",engine="tensorrt",batch_size="16",le="0.01"} 500
  benchlab_runner_inference_latency_seconds_sum{model="resnet50",engine="tensorrt",batch_size="16"} 4.5
  ```

#### `benchlab_runner_model_load_duration_seconds`
- **Type:** Histogram
- **Description:** Duration to load model from disk or build from scratch
- **Labels:**
  - `model`: Model name
  - `engine`: Engine type
- **Buckets:** [0.1, 0.5, 1, 5, 10, 30, 60]
- **Example:**
  ```
  benchlab_runner_model_load_duration_seconds_sum{model="resnet50",engine="pytorch_cuda"} 15.3
  benchlab_runner_model_load_duration_seconds_count{model="resnet50",engine="pytorch_cuda"} 10
  ```

#### `benchlab_runner_engine_build_duration_seconds`
- **Type:** Histogram
- **Description:** Duration to build TensorRT engine (ONNX → TRT compilation)
- **Labels:**
  - `model`: Model name
  - `batch_size`: Max batch size in optimization profile
  - `precision`: Engine precision ("fp16", "fp32")
- **Buckets:** [1, 5, 10, 30, 60, 120, 300]
- **Example:**
  ```
  benchlab_runner_engine_build_duration_seconds_sum{model="resnet50",batch_size="16",precision="fp16"} 45.2
  benchlab_runner_engine_build_duration_seconds_count{model="resnet50",batch_size="16",precision="fp16"} 2
  ```

### GPU Telemetry Metrics

#### `benchlab_runner_gpu_utilization_percent`
- **Type:** Gauge
- **Description:** Current GPU utilization percentage (0-100)
- **Updated:** On each /metrics scrape
- **Example:**
  ```
  benchlab_runner_gpu_utilization_percent 85.5
  ```

#### `benchlab_runner_gpu_memory_used_bytes`
- **Type:** Gauge
- **Description:** Current GPU memory used in bytes
- **Updated:** On each /metrics scrape
- **Example:**
  ```
  benchlab_runner_gpu_memory_used_bytes 8589934592  # 8 GB
  ```

#### `benchlab_runner_gpu_temperature_celsius`
- **Type:** Gauge
- **Description:** Current GPU temperature in Celsius
- **Updated:** On each /metrics scrape
- **Example:**
  ```
  benchlab_runner_gpu_temperature_celsius 72.0
  ```

#### `benchlab_runner_gpu_power_usage_watts`
- **Type:** Gauge
- **Description:** Current GPU power usage in watts
- **Updated:** On each /metrics scrape
- **Example:**
  ```
  benchlab_runner_gpu_power_usage_watts 250.5
  ```

### Runner Info

#### `benchlab_runner_info`
- **Type:** Info
- **Description:** Runner version and configuration information
- **Labels:** `version`, `service`
- **Example:**
  ```
  benchlab_runner_info{version="0.1.0",service="tensorrt-benchlab-runner"} 1
  ```

---

## Metric Types

### Counter
- **Monotonically increasing** value
- Never decreases (except on restart)
- Use for counting events (total requests, errors, etc.)
- Query with `rate()` or `increase()` in PromQL

### Gauge
- **Point-in-time** measurement
- Can increase or decrease
- Use for current state (active connections, memory usage, temperature)
- Query directly or with `avg_over_time()`

### Histogram
- **Distribution** of values
- Provides `_sum`, `_count`, and `_bucket` metrics
- Use for latencies, durations, sizes
- Query with `histogram_quantile()` for percentiles

### Info
- **Metadata** about the service
- Always has value 1
- Use for version information, static config

---

## Usage Examples

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'benchlab-controller'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'benchlab-runner'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### PromQL Queries

#### Controller Queries

**Run creation rate (last 5 minutes):**
```promql
rate(benchlab_controller_runs_total[5m])
```

**Average run duration (last hour):**
```promql
rate(benchlab_controller_run_duration_seconds_sum[1h]) /
rate(benchlab_controller_run_duration_seconds_count[1h])
```

**P95 run duration:**
```promql
histogram_quantile(0.95,
  rate(benchlab_controller_run_duration_seconds_bucket[5m]))
```

**Current active runs:**
```promql
benchlab_controller_active_runs
```

**Success rate (last 1 hour):**
```promql
sum(rate(benchlab_controller_runs_total{status="succeeded"}[1h])) /
sum(rate(benchlab_controller_runs_total[1h]))
```

**Runner request retry rate:**
```promql
sum(rate(benchlab_controller_runner_request_retries_total[5m])) by (engine)
```

#### Runner Queries

**Benchmark execution rate:**
```promql
rate(benchlab_runner_benchmark_executions_total[5m])
```

**P50 inference latency by engine and batch size:**
```promql
histogram_quantile(0.50,
  rate(benchlab_runner_inference_latency_seconds_bucket[5m]))
```

**P99 inference latency for TensorRT:**
```promql
histogram_quantile(0.99,
  rate(benchlab_runner_inference_latency_seconds_bucket{engine="tensorrt"}[5m]))
```

**Average GPU utilization (last 10 minutes):**
```promql
avg_over_time(benchlab_runner_gpu_utilization_percent[10m])
```

**Max GPU temperature (last 1 hour):**
```promql
max_over_time(benchlab_runner_gpu_temperature_celsius[1h])
```

**GPU memory usage percentage:**
```promql
(benchlab_runner_gpu_memory_used_bytes /
 (benchlab_runner_gpu_memory_used_bytes +
  (on() benchlab_runner_gpu_memory_total_bytes - benchlab_runner_gpu_memory_used_bytes))) * 100
```

**TensorRT engine build time P95:**
```promql
histogram_quantile(0.95,
  rate(benchlab_runner_engine_build_duration_seconds_bucket{precision="fp16"}[1h]))
```

### Grafana Dashboard Panels

#### Run Status Overview
```promql
# Pie chart or table
benchlab_controller_runs_by_status
```

#### Benchmark Throughput
```promql
# Graph: runs per minute
sum(rate(benchlab_controller_runs_total[1m])) by (model)
```

#### Inference Latency Heatmap
```promql
# Heatmap
sum(rate(benchlab_runner_inference_latency_seconds_bucket[5m])) by (le, engine)
```

#### GPU Utilization Timeline
```promql
# Graph
benchlab_runner_gpu_utilization_percent
```

### Alerting Rules

```yaml
groups:
  - name: benchlab_alerts
    rules:
      # Alert if run failure rate > 10%
      - alert: HighRunFailureRate
        expr: |
          sum(rate(benchlab_controller_runs_total{status="failed"}[5m])) /
          sum(rate(benchlab_controller_runs_total[5m])) > 0.1
        for: 5m
        annotations:
          summary: "High benchmark run failure rate"

      # Alert if GPU temperature > 85°C
      - alert: HighGPUTemperature
        expr: benchlab_runner_gpu_temperature_celsius > 85
        for: 2m
        annotations:
          summary: "GPU temperature critical"

      # Alert if runner requests are timing out frequently
      - alert: RunnerRequestTimeouts
        expr: |
          rate(benchlab_controller_runner_request_retries_total{attempt="3"}[5m]) > 0.5
        for: 5m
        annotations:
          summary: "Frequent runner request timeouts"
```

---

## Best Practices

1. **Scrape Interval:** Use 15-30 second intervals for most metrics
2. **Retention:** Keep detailed metrics for 15 days, aggregated for longer
3. **Labels:** Keep cardinality low - avoid high-cardinality labels like run_id
4. **Histograms:** Choose bucket boundaries based on expected latency ranges
5. **Grafana:** Create dashboards for operational visibility
6. **Alerts:** Set up alerts for failure rates, timeouts, and resource usage

---

## Monitoring Stack Setup

### Docker Compose Example

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  prometheus_data:
  grafana_data:
```

### Access

- **Prometheus UI:** http://localhost:9090
- **Grafana UI:** http://localhost:3000
- **Metrics:**
  - Controller: http://localhost:8000/metrics
  - Runner: http://localhost:8001/metrics

---

## Troubleshooting

### Metrics Not Appearing

1. **Check endpoint accessibility:**
   ```bash
   curl http://localhost:8000/metrics
   curl http://localhost:8001/metrics
   ```

2. **Verify Prometheus scrape configuration**
3. **Check Prometheus targets:** http://localhost:9090/targets

### Incorrect Values

1. **Counter reset:** Counters reset on service restart (use `rate()` or `increase()`)
2. **Label mismatch:** Ensure labels match exactly
3. **Time range:** Adjust query time range for sufficient data

### High Cardinality

- **Symptom:** Slow Prometheus queries, high memory usage
- **Cause:** Too many unique label combinations
- **Fix:** Avoid labels with many unique values (e.g., run_id, timestamp)

---

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [PromQL Guide](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [prometheus_client Python Library](https://github.com/prometheus/client_python)
