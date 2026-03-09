"""Tests for Pydantic schema validation."""
import pytest
from datetime import datetime
from app.schemas.bench import BenchmarkRequest, EngineType


class TestBenchmarkRequest:
    """Test suite for BenchmarkRequest schema."""

    def test_default_values(self):
        """Test that default values are applied correctly."""
        request = BenchmarkRequest(
            model_name="resnet50",
            engine_type=EngineType.PYTORCH_CPU
        )

        # Updated defaults
        assert request.batch_sizes == [1, 4, 8, 16]
        assert request.num_iterations == 50
        assert request.warmup_iterations == 10
