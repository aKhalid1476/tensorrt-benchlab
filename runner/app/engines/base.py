"""Base inference engine interface.

This module defines the abstract interface that all inference engines must implement,
ensuring consistent behavior across PyTorch CPU, PyTorch GPU, and TensorRT engines.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import torch


class InferenceEngine(ABC):
    """
    Abstract base class for inference engines.

    All engines must implement:
    - load_model(): Load model weights and prepare for inference
    - infer(batch): Run inference on a batch and return latency
    - name: Engine name (property)
    - device: Device name (property)
    - metadata(): Return engine-specific metadata
    """

    def __init__(self, model_name: str):
        """
        Initialize inference engine.

        Args:
            model_name: Model identifier (e.g., 'resnet50', 'mobilenet_v2')
        """
        self.model_name = model_name
        self.model = None
        self.input_shape: Tuple[int, ...] = (3, 224, 224)  # ImageNet default (C, H, W)
        self._loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model into memory and prepare for inference.

        This method should:
        1. Load pretrained weights
        2. Set model to eval() mode
        3. Move model to appropriate device
        4. Set _loaded flag to True
        """
        pass

    @abstractmethod
    def infer(self, inputs: torch.Tensor) -> float:
        """
        Run inference on a batch and return latency.

        Args:
            inputs: Input tensor with shape (batch_size, *input_shape)

        Returns:
            Latency in seconds (float)

        Raises:
            RuntimeError: If model not loaded
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable engine name."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return device name (e.g., 'cpu', 'cuda:0')."""
        pass

    def metadata(self) -> Dict[str, Any]:
        """
        Return engine-specific metadata.

        Returns:
            Dictionary with engine information
        """
        return {
            "engine_name": self.name,
            "model_name": self.model_name,
            "device": self.device,
            "input_shape": self.input_shape,
            "loaded": self._loaded,
        }

    def __repr__(self) -> str:
        """String representation of engine."""
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"
