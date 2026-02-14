"""PyTorch CPU inference engine."""
import time

import torch
import torchvision.models as models

from .base import InferenceEngine


class TorchCPUEngine(InferenceEngine):
    """PyTorch CPU inference engine."""

    def load_model(self) -> None:
        """Load pretrained model for CPU inference."""
        # Load pretrained model
        if self.model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model.eval()
        self.model.to("cpu")

    def infer(self, inputs: torch.Tensor) -> float:
        """Run inference on CPU."""
        inputs = inputs.to("cpu")

        start = time.perf_counter()
        with torch.no_grad():
            _ = self.model(inputs)
        end = time.perf_counter()

        return end - start
