"""PyTorch CUDA GPU inference engine.

This module implements GPU-based inference using PyTorch with CUDA.
Preprocessing is identical to CPU engine. Proper CUDA synchronization
ensures accurate timing measurements.
"""
import logging
import time
from typing import Dict, Any

import torch
import torchvision.models as models

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class TorchCUDAEngine(InferenceEngine):
    """
    PyTorch CUDA GPU inference engine.

    Uses torchvision pretrained models with GPU acceleration.
    Identical preprocessing to CPU engine for fair comparison.
    Proper CUDA synchronization for accurate timing.
    """

    def __init__(self, model_name: str):
        """Initialize CUDA engine."""
        super().__init__(model_name)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self._device = "cuda"

    def load_model(self) -> None:
        """
        Load pretrained model for CUDA inference.

        Loads same torchvision pretrained weights as CPU engine,
        moves model to GPU, and sets eval mode.
        """
        logger.info(f"event=load_model engine=torch_cuda model={self.model_name}")

        # Load pretrained model (identical to CPU engine)
        if self.model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Set to eval mode (identical to CPU engine)
        self.model.eval()

        # Move to CUDA
        self.model.to(self._device)

        self._loaded = True
        logger.info(
            f"event=model_loaded engine=torch_cuda model={self.model_name} "
            f"device={torch.cuda.get_device_name(0)}"
        )

    def infer(self, inputs: torch.Tensor) -> float:
        """
        Run inference on CUDA GPU with proper synchronization.

        IMPORTANT: CUDA operations are asynchronous. We must synchronize
        before and after inference to get accurate timing.

        Args:
            inputs: Input tensor with shape (batch_size, 3, 224, 224)

        Returns:
            Latency in seconds

        Raises:
            RuntimeError: If model not loaded
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure inputs are on CUDA
        inputs = inputs.to(self._device)

        # CRITICAL: Synchronize before timing to wait for any pending operations
        torch.cuda.synchronize()

        # Start timing
        start = time.perf_counter()

        # Run inference with gradients disabled (identical to CPU)
        with torch.no_grad():
            _ = self.model(inputs)

        # CRITICAL: Synchronize after inference to wait for GPU completion
        torch.cuda.synchronize()

        # End timing
        end = time.perf_counter()

        return end - start

    @property
    def name(self) -> str:
        """Return engine name."""
        return "PyTorch CUDA"

    @property
    def device(self) -> str:
        """Return device name."""
        return self._device

    def metadata(self) -> Dict[str, Any]:
        """Return CUDA engine metadata."""
        meta = super().metadata()
        meta.update({
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        })
        return meta
