# Benchmark Methodology

## Core Configuration

**Defaults:**
- Warmup iterations: **10** (excluded from statistics)
- Measurement iterations: **50**
- Batch sizes: **[1, 4, 8, 16]**
- Random seed: **42** (for reproducibility)

## Statistics Computed

- **p50** (median): 50th percentile latency
- **p95**: 95th percentile latency  
- **mean**: Average latency
- **stddev**: Standard deviation (sample, ddof=1)
- **throughput**: batch_size / mean_latency (req/s)

## Workflow

1. **Model Loading**: Load pretrained weights
2. **Input Generation**: Fixed random inputs (seed=42)
3. **Warmup Phase**: Run 10 iterations, discard results
4. **Measurement Phase**: Run 50 iterations, collect latencies
5. **Statistics**: Calculate p50, p95, mean, stddev, throughput

## Reproducibility

✅ Fixed random seed (42)
✅ Warmup excluded from stats
✅ Deterministic calculations
✅ Environment metadata captured

