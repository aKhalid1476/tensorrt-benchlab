"""Benchmark methodology and statistics calculation."""
import numpy as np

from ..schemas.bench import BatchStats


def calculate_batch_statistics(batch_size: int, latencies: list[float]) -> BatchStats:
    """
    Calculate statistical metrics for a batch size run.

    Args:
        batch_size: Batch size tested
        latencies: List of latency measurements in seconds

    Returns:
        BatchStats with calculated metrics
    """
    latencies_ms = np.array(latencies) * 1000  # Convert to milliseconds

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    mean = float(np.mean(latencies_ms))
    stddev = float(np.std(latencies_ms))

    # Throughput = batch_size / mean_latency_seconds
    mean_latency_sec = mean / 1000
    throughput = batch_size / mean_latency_sec if mean_latency_sec > 0 else 0.0

    return BatchStats(
        batch_size=batch_size,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_mean_ms=mean,
        latency_stddev_ms=stddev,
        throughput_req_per_sec=throughput,
    )
