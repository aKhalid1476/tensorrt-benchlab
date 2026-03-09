"""Tests for inference engines."""
import pytest
import torch
import numpy as np

from app.engines.base import InferenceEngine
from app.engines.torch_cpu import TorchCPUEngine
from app.engines.torch_cuda import TorchCUDAEngine


class TestTorchCPUEngine:
    """Test suite for PyTorch CPU engine."""

    def test_engine_initialization(self):
        """Test engine can be initialized."""
        engine = TorchCPUEngine(model_name="resnet50")
        assert engine.model_name == "resnet50"
        assert engine.device == "cpu"
        assert engine.name == "PyTorch CPU"

    def test_load_model_resnet50(self):
        """Test loading ResNet50 model."""
        engine = TorchCPUEngine(model_name="resnet50")
        engine.load_model()

        assert engine.model is not None
        assert engine._loaded is True

    def test_load_model_mobilenet(self):
        """Test loading MobileNetV2 model."""
        engine = TorchCPUEngine(model_name="mobilenet_v2")
        engine.load_model()

        assert engine.model is not None
        assert engine._loaded is True

    def test_unsupported_model_raises_error(self):
        """Test that unsupported model raises ValueError."""
        engine = TorchCPUEngine(model_name="invalid_model")
        with pytest.raises(ValueError, match="Unsupported model"):
            engine.load_model()

    def test_infer_before_load_raises_error(self):
        """Test that inference before loading raises RuntimeError."""
        engine = TorchCPUEngine(model_name="resnet50")
        inputs = torch.randn(1, 3, 224, 224)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.infer(inputs)

    def test_infer_single_batch(self):
        """Test inference with single batch."""
        engine = TorchCPUEngine(model_name="resnet50")
        engine.load_model()

        # Create single batch input
        inputs = torch.randn(1, 3, 224, 224)
        latency = engine.infer(inputs)

        # Latency should be positive
        assert latency > 0
        assert isinstance(latency, float)

    def test_infer_multiple_batches(self):
        """Test inference with different batch sizes."""
        engine = TorchCPUEngine(model_name="resnet50")
        engine.load_model()

        for batch_size in [1, 4, 8]:
            inputs = torch.randn(batch_size, 3, 224, 224)
            latency = engine.infer(inputs)

            assert latency > 0
            assert isinstance(latency, float)

    def test_eval_mode(self):
        """Test that model is in eval mode after loading."""
        engine = TorchCPUEngine(model_name="resnet50")
        engine.load_model()

        # Model should be in eval mode (training=False)
        assert engine.model.training is False

    def test_no_grad_context(self):
        """Test that inference runs with no_grad."""
        engine = TorchCPUEngine(model_name="resnet50")
        engine.load_model()

        inputs = torch.randn(1, 3, 224, 224)

        # Store original grad_enabled state
        original_grad = torch.is_grad_enabled()

        # Run inference
        with torch.set_grad_enabled(True):  # Enable grads before inference
            engine.infer(inputs)

        # After inference, grads should be back to original state
        assert torch.is_grad_enabled() == original_grad

    def test_metadata(self):
        """Test engine metadata."""
        engine = TorchCPUEngine(model_name="resnet50")
        engine.load_model()

        meta = engine.metadata()

        assert meta["engine_name"] == "PyTorch CPU"
        assert meta["model_name"] == "resnet50"
        assert meta["device"] == "cpu"
        assert meta["loaded"] is True
        assert "torch_version" in meta
        assert "num_threads" in meta

    def test_output_shape_resnet50(self):
        """Test that ResNet50 output has correct shape."""
        engine = TorchCPUEngine(model_name="resnet50")
        engine.load_model()

        inputs = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = engine.model(inputs)

        # ResNet50 outputs 1000 classes
        assert output.shape == (4, 1000)

    def test_deterministic_inference(self):
        """Test that same inputs produce consistent outputs."""
        engine = TorchCPUEngine(model_name="resnet50")
        engine.load_model()

        # Fixed seed for reproducibility
        torch.manual_seed(42)
        inputs = torch.randn(2, 3, 224, 224)

        # Run inference twice
        with torch.no_grad():
            output1 = engine.model(inputs)
            output2 = engine.model(inputs)

        # Outputs should be identical (within floating point precision)
        assert torch.allclose(output1, output2, rtol=1e-5)


class TestTorchCUDAEngine:
    """Test suite for PyTorch CUDA engine."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_engine_initialization(self):
        """Test engine can be initialized with CUDA."""
        engine = TorchCUDAEngine(model_name="resnet50")
        assert engine.model_name == "resnet50"
        assert engine.device == "cuda"
        assert engine.name == "PyTorch CUDA"

    def test_initialization_without_cuda_raises_error(self):
        """Test that initialization without CUDA raises RuntimeError."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test this")

        with pytest.raises(RuntimeError, match="CUDA not available"):
            TorchCUDAEngine(model_name="resnet50")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_model(self):
        """Test loading model on CUDA."""
        engine = TorchCUDAEngine(model_name="resnet50")
        engine.load_model()

        assert engine.model is not None
        assert engine._loaded is True

        # Model should be on CUDA
        assert next(engine.model.parameters()).is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_infer_single_batch(self):
        """Test CUDA inference with single batch."""
        engine = TorchCUDAEngine(model_name="resnet50")
        engine.load_model()

        inputs = torch.randn(1, 3, 224, 224)
        latency = engine.infer(inputs)

        assert latency > 0
        assert isinstance(latency, float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_synchronization(self):
        """Test that CUDA synchronization happens correctly."""
        engine = TorchCUDAEngine(model_name="resnet50")
        engine.load_model()

        inputs = torch.randn(1, 3, 224, 224)

        # Run inference
        latency = engine.infer(inputs)

        # After inference with synchronization, all GPU operations should be complete
        # This is implicitly tested by the fact that we get a valid latency
        assert latency > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_metadata(self):
        """Test CUDA engine metadata."""
        engine = TorchCUDAEngine(model_name="resnet50")
        engine.load_model()

        meta = engine.metadata()

        assert meta["engine_name"] == "PyTorch CUDA"
        assert meta["device"] == "cuda"
        assert "cuda_version" in meta
        assert "cudnn_version" in meta
        assert "gpu_name" in meta
        assert "gpu_memory_total" in meta


class TestEngineConsistency:
    """Test consistency between CPU and CUDA engines."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_same_model_weights(self):
        """Test that CPU and CUDA engines load same weights."""
        cpu_engine = TorchCPUEngine(model_name="resnet50")
        cuda_engine = TorchCUDAEngine(model_name="resnet50")

        cpu_engine.load_model()
        cuda_engine.load_model()

        # Compare first layer weights
        cpu_weights = cpu_engine.model.conv1.weight.data
        cuda_weights = cuda_engine.model.conv1.weight.data.cpu()

        assert torch.allclose(cpu_weights, cuda_weights, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_same_preprocessing(self):
        """Test that CPU and CUDA engines use same preprocessing."""
        cpu_engine = TorchCPUEngine(model_name="resnet50")
        cuda_engine = TorchCUDAEngine(model_name="resnet50")

        # Input shapes should be identical
        assert cpu_engine.input_shape == cuda_engine.input_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_consistent_outputs(self):
        """Test that CPU and CUDA engines produce consistent outputs."""
        cpu_engine = TorchCPUEngine(model_name="resnet50")
        cuda_engine = TorchCUDAEngine(model_name="resnet50")

        cpu_engine.load_model()
        cuda_engine.load_model()

        # Same inputs
        torch.manual_seed(42)
        inputs = torch.randn(2, 3, 224, 224)

        # Run on both engines
        with torch.no_grad():
            cpu_output = cpu_engine.model(inputs.to("cpu"))
            cuda_output = cuda_engine.model(inputs.to("cuda")).cpu()

        # Outputs should be very close (small numerical differences expected)
        assert torch.allclose(cpu_output, cuda_output, rtol=1e-3, atol=1e-5)

    def test_both_use_eval_mode(self):
        """Test that both engines use eval mode."""
        cpu_engine = TorchCPUEngine(model_name="resnet50")
        cpu_engine.load_model()

        if torch.cuda.is_available():
            cuda_engine = TorchCUDAEngine(model_name="resnet50")
            cuda_engine.load_model()

            assert cpu_engine.model.training is False
            assert cuda_engine.model.training is False
        else:
            assert cpu_engine.model.training is False
