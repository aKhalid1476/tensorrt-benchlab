"""PyTorch CPU inference engine.

This module implements CPU-based inference using PyTorch with torchvision models.
Preprocessing is standardized to match GPU and TensorRT implementations.
"""
import logging
import time
from typing import Dict, Any

import torch
import torchvision.models as models

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class TorchCPUEngine(InferenceEngine):
    """
    PyTorch CPU inference engine.

    Uses torchvision pretrained models with standardized preprocessing.
    All inference runs in eval() mode with torch.no_grad() for consistency.
    """

    def __init__(self, model_name: str):
        """Initialize CPU engine."""
        super().__init__(model_name)
        self._device = "cpu"

    def load_model(self) -> None:
        """
        Load pretrained model for CPU inference.

        Loads torchvision pretrained weights and sets model to eval mode.
        """
        logger.info(f"event=load_model engine=torch_cpu model={self.model_name}")

        # Load pretrained model
        if self.model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Set to eval mode (disables dropout, batch norm updates, etc.)
        self.model.eval()

        # Move to CPU
        self.model.to(self._device)

        self._loaded = True
        logger.info(f"event=model_loaded engine=torch_cpu model={self.model_name}")

    def infer(self, inputs: torch.Tensor) -> float:
        """
        Run inference on CPU.

        Args:
            inputs: Input tensor with shape (batch_size, 3, 224, 224)

        Returns:
            Latency in seconds

        Raises:
            RuntimeError: If model not loaded
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure inputs are on CPU
        inputs = inputs.to(self._device)

        # Measure inference time
        start = time.perf_counter()
        with torch.no_grad():  # Disable gradient computation
            _ = self.model(inputs)
        end = time.perf_counter()

        return end - start

    @property
    def name(self) -> str:
        """Return engine name."""
        return "PyTorch CPU"

    @property
    def device(self) -> str:
        """Return device name."""
        return self._device

    def metadata(self) -> Dict[str, Any]:
        """Return CPU engine metadata."""
        meta = super().metadata()
        meta.update({
            "torch_version": torch.__version__,
            "num_threads": torch.get_num_threads(),
        })
        return meta
