"""Benchmark methodology and statistics calculation.

This module implements the core benchmarking methodology ensuring
reproducible and statistically sound measurements.

Methodology:
1. Fixed random seed (42) for input generation
2. Warmup iterations (default: 10) to stabilize system state
3. Measured iterations (default: 50) with precise timing
4. Statistical analysis: p50, p95, mean, stddev, throughput
5. Warmup results excluded from statistics
"""
import logging
from typing import List

import numpy as np

from ..schemas.bench import BatchStats

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_MEASUREMENT_ITERATIONS = 50
DEFAULT_BATCH_SIZES = [1, 4, 8, 16]
RANDOM_SEED = 42


def calculate_batch_statistics(batch_size: int, latencies: List[float]) -> BatchStats:
    """
    Calculate statistical metrics for a batch size run.

    This function computes percentiles, mean, standard deviation, and throughput
    from a list of latency measurements. All warmup iterations should be excluded
    before calling this function.

    Args:
        batch_size: Number of samples per batch
        latencies: List of latency measurements in seconds (warmup excluded)

    Returns:
        BatchStats with calculated metrics

    Raises:
        ValueError: If latencies list is empty or contains invalid values

    Example:
        >>> latencies = [0.01, 0.012, 0.011, 0.013, 0.01]  # seconds
        >>> stats = calculate_batch_statistics(batch_size=4, latencies=latencies)
        >>> print(f"p50: {stats.latency_p50_ms:.2f}ms")
        p50: 11.00ms
    """
    if not latencies:
        raise ValueError("Latencies list cannot be empty")

    if any(lat <= 0 for lat in latencies):
        raise ValueError("All latencies must be positive")

    # Convert to numpy array and milliseconds
    latencies_ms = np.array(latencies, dtype=np.float64) * 1000.0

    # Calculate percentiles using linear interpolation (NumPy default)
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))

    # Calculate mean and standard deviation
    mean = float(np.mean(latencies_ms))
    stddev = float(np.std(latencies_ms, ddof=1))  # Sample std dev (ddof=1)

    # Throughput calculation
    # throughput (req/s) = batch_size / mean_latency (seconds)
    mean_latency_sec = mean / 1000.0
    throughput = batch_size / mean_latency_sec if mean_latency_sec > 0 else 0.0

    logger.debug(
        f"event=stats_calculated batch_size={batch_size} n_samples={len(latencies)} "
        f"p50={p50:.2f}ms p95={p95:.2f}ms mean={mean:.2f}ms stddev={stddev:.2f}ms "
        f"throughput={throughput:.2f}req/s"
    )

    return BatchStats(
        batch_size=batch_size,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_mean_ms=mean,
        latency_stddev_ms=stddev,
        throughput_req_per_sec=throughput,
    )


def validate_benchmark_config(
    batch_sizes: List[int],
    num_iterations: int,
    warmup_iterations: int
) -> None:
    """
    Validate benchmark configuration parameters.

    Args:
        batch_sizes: List of batch sizes to test
        num_iterations: Number of measured iterations
        warmup_iterations: Number of warmup iterations

    Raises:
        ValueError: If configuration is invalid
    """
    if not batch_sizes:
        raise ValueError("batch_sizes cannot be empty")

    if any(bs <= 0 for bs in batch_sizes):
        raise ValueError("All batch sizes must be positive")

    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")

    # Warn if configuration seems unusual
    if num_iterations < 10:
        logger.warning(
            f"event=low_iterations num_iterations={num_iterations} "
            "message='Low iteration count may produce unreliable statistics'"
        )

    if warmup_iterations == 0:
        logger.warning(
            "event=no_warmup message='Zero warmup iterations may include cold-start effects'"
        )
