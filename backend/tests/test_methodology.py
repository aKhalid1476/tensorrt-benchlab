"""Tests for benchmark methodology calculations."""
import pytest
import numpy as np
from app.bench.methodology import (
    calculate_batch_statistics,
    validate_benchmark_config,
    DEFAULT_WARMUP_ITERATIONS,
    DEFAULT_MEASUREMENT_ITERATIONS,
)
from app.schemas.bench import BatchStats


class TestCalculateBatchStatistics:
    """Test suite for calculate_batch_statistics function."""

    def test_basic_calculation(self):
        """Test basic statistical calculations."""
        latencies = [0.01, 0.012, 0.011, 0.013, 0.01, 0.015, 0.011, 0.012, 0.01, 0.014]

        stats = calculate_batch_statistics(batch_size=4, latencies=latencies)

        assert stats.batch_size == 4
        assert stats.latency_p50_ms > 0
        assert stats.latency_p95_ms >= stats.latency_p50_ms
        assert stats.latency_mean_ms > 0
        assert stats.latency_stddev_ms >= 0
        assert stats.throughput_req_per_sec > 0

    def test_throughput_calculation(self):
        """Test throughput calculation formula."""
        # All latencies = 10ms = 0.01s
        latencies = [0.01] * 10

        stats = calculate_batch_statistics(batch_size=8, latencies=latencies)

        # With 10ms latency and batch_size 8:
        # throughput = 8 / 0.01 = 800 req/s
        expected_throughput = 8 / 0.01
        assert abs(stats.throughput_req_per_sec - expected_throughput) < 1.0

    def test_deterministic_output(self):
        """Test that same inputs produce same outputs (determinism)."""
        latencies = [0.01 + i * 0.001 for i in range(20)]

        stats1 = calculate_batch_statistics(batch_size=4, latencies=latencies)
        stats2 = calculate_batch_statistics(batch_size=4, latencies=latencies)

        # All fields should be exactly equal
        assert stats1.model_dump() == stats2.model_dump()
