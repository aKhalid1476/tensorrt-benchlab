"""Tests for TensorRT engine."""
import pytest
import torch

from app.engines.tensorrt import TensorRTEngine, TENSORRT_AVAILABLE


class TestTensorRTEngine:
    """Test suite for TensorRT engine."""

    def test_engine_initialization(self):
        """Test engine can be initialized."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="resnet50")
        assert engine.model_name == "resnet50"
        assert engine.device == "cuda"

    def test_initialization_without_cuda_raises_error(self):
        """Test that initialization without CUDA raises RuntimeError."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test this")

        with pytest.raises(RuntimeError, match="CUDA not available"):
            TensorRTEngine(model_name="resnet50")

    def test_load_model(self):
        """Test model loading (ONNX export or fallback)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="resnet50")
        engine.load_model()

        assert engine._loaded is True

        # Should have either TensorRT engine or fallback model
        if TENSORRT_AVAILABLE:
            # ONNX should be exported
            onnx_path = engine.cache_dir / "resnet50.onnx"
            # Note: ONNX will be exported on first load
        else:
            assert engine.fallback_model is not None

    def test_fallback_mode_when_tensorrt_unavailable(self):
        """Test that fallback works when TensorRT unavailable."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="resnet50")

        if not TENSORRT_AVAILABLE:
            assert engine.use_fallback is True
            assert engine.name == "TensorRT (PyTorch Fallback)"
        else:
            assert engine.use_fallback is False
            assert engine.name == "TensorRT"

    def test_infer_single_batch(self):
        """Test inference with single batch."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="resnet50")
        engine.load_model()

        inputs = torch.randn(1, 3, 224, 224)
        latency = engine.infer(inputs)

        assert latency > 0
        assert isinstance(latency, float)

    def test_infer_multiple_batch_sizes(self):
        """Test inference with different batch sizes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="resnet50")
        engine.load_model()

        for batch_size in [1, 4]:
            inputs = torch.randn(batch_size, 3, 224, 224)
            latency = engine.infer(inputs)

            assert latency > 0
            assert isinstance(latency, float)

    def test_infer_before_load_raises_error(self):
        """Test that inference before loading raises RuntimeError."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="resnet50")
        inputs = torch.randn(1, 3, 224, 224)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.infer(inputs)

    def test_unsupported_model_raises_error(self):
        """Test that unsupported model raises ValueError."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="invalid_model")
        with pytest.raises(ValueError, match="Unsupported model"):
            engine.load_model()

    def test_metadata(self):
        """Test engine metadata."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="resnet50")
        engine.load_model()

        meta = engine.metadata()

        assert "engine_name" in meta
        assert meta["model_name"] == "resnet50"
        assert meta["device"] == "cuda"
        assert "tensorrt_available" in meta
        assert "using_fallback" in meta
        assert meta["loaded"] is True

    @pytest.mark.skipif(not TENSORRT_AVAILABLE, reason="TensorRT not available")
    def test_onnx_export_caching(self):
        """Test that ONNX export is cached."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="resnet50")
        engine.load_model()

        onnx_path = engine.cache_dir / "resnet50.onnx"
        assert onnx_path.exists()

        # Check file size
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        assert size_mb > 0  # Should have content

    @pytest.mark.skipif(not TENSORRT_AVAILABLE, reason="TensorRT not available")
    def test_engine_caching_per_batch_size(self):
        """Test that TensorRT engines are cached per batch size."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        engine = TensorRTEngine(model_name="resnet50")
        engine.load_model()

        # Run inference with different batch sizes
        for batch_size in [1, 4]:
            inputs = torch.randn(batch_size, 3, 224, 224)
            engine.infer(inputs)

        # Check that engines were cached
        meta = engine.metadata()
        cached = meta.get("cached_engines", [])
        # At least one batch size should be cached
        assert len(cached) > 0

    def test_fallback_uses_pytorch_cuda(self):
        """Test that fallback uses PyTorch CUDA engine."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if TENSORRT_AVAILABLE:
            pytest.skip("TensorRT is available, fallback not used")

        engine = TensorRTEngine(model_name="resnet50")
        engine.load_model()

        assert engine.fallback_model is not None
        # Model should be on CUDA
        assert next(engine.fallback_model.parameters()).is_cuda
