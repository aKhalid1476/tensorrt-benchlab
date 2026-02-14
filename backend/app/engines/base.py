"""Base inference engine interface."""
from abc import ABC, abstractmethod
from typing import Tuple

import torch


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.input_shape: Tuple[int, ...] = (3, 224, 224)  # Default ImageNet shape

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def infer(self, inputs: torch.Tensor) -> float:
        """
        Run inference and return latency.

        Args:
            inputs: Input tensor

        Returns:
            Latency in seconds
        """
        pass
