"""PyTorch CUDA GPU inference engine."""
import time

import torch
import torchvision.models as models

from .base import InferenceEngine


class TorchCUDAEngine(InferenceEngine):
    """PyTorch CUDA GPU inference engine."""

    def load_model(self) -> None:
        """Load pretrained model for CUDA inference."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # Load pretrained model
        if self.model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model.eval()
        self.model.to("cuda")

    def infer(self, inputs: torch.Tensor) -> float:
        """Run inference on CUDA GPU."""
        inputs = inputs.to("cuda")

        # Synchronize before timing
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = self.model(inputs)

        # Synchronize after inference
        torch.cuda.synchronize()
        end = time.perf_counter()

        return end - start
