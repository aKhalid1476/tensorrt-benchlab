"""Tests for benchmark methodology calculations."""
import pytest
from app.bench.methodology import calculate_batch_statistics


def test_calculate_batch_statistics():
    """Test statistical calculations."""
    latencies = [0.01, 0.012, 0.011, 0.013, 0.01, 0.015, 0.011, 0.012, 0.01, 0.014]
    
    stats = calculate_batch_statistics(batch_size=4, latencies=latencies)
    
    assert stats.batch_size == 4
    assert stats.latency_p50_ms > 0
    assert stats.latency_p95_ms >= stats.latency_p50_ms
    assert stats.latency_mean_ms > 0
    assert stats.latency_stddev_ms >= 0
    assert stats.throughput_req_per_sec > 0


def test_throughput_calculation():
    """Test throughput calculation."""
    latencies = [0.01] * 10  # 10ms each
    
    stats = calculate_batch_statistics(batch_size=8, latencies=latencies)
    
    # With 10ms latency and batch_size 8, throughput should be 800 req/s
    expected_throughput = 8 / 0.01
    assert abs(stats.throughput_req_per_sec - expected_throughput) < 1.0
