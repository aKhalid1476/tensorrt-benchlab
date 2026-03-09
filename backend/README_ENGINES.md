# Inference Engines

## Quick Start

```python
from app.engines.torch_cpu import TorchCPUEngine

# Initialize engine
engine = TorchCPUEngine(model_name="resnet50")

# Load model
engine.load_model()

# Prepare inputs
import torch
inputs = torch.randn(4, 3, 224, 224)  # batch_size=4

# Run inference
latency = engine.infer(inputs)  # Returns latency in seconds
print(f"Latency: {latency*1000:.2f}ms")

# Get metadata
meta = engine.metadata()
print(f"Engine: {meta['engine_name']}")
print(f"Device: {meta['device']}")
```

## Available Engines

| Engine | Device | Models | Status |
|--------|--------|--------|--------|
| `TorchCPUEngine` | CPU | ResNet50, MobileNetV2 | ✅ Ready |
| `TorchCUDAEngine` | CUDA GPU | ResNet50, MobileNetV2 | ✅ Ready |
| `TensorRTEngine` | CUDA GPU | - | 🚧 Placeholder |

## Supported Models

- **resnet50**: ResNet-50 (25.6M parameters, 1000 classes)
- **mobilenet_v2**: MobileNet V2 (3.5M parameters, 1000 classes)

## Interface

All engines implement:

```python
class InferenceEngine:
    def load_model() -> None
    def infer(inputs: Tensor) -> float
    @property name -> str
    @property device -> str
    def metadata() -> Dict[str, Any]
```

## Testing

```bash
# Run all engine tests
pytest tests/test_engines.py -v

# Run specific engine tests
pytest tests/test_engines.py::TestTorchCPUEngine -v

# Run CUDA tests (requires GPU)
pytest tests/test_engines.py::TestTorchCUDAEngine -v
```

## Adding New Models

Edit both `torch_cpu.py` and `torch_cuda.py`:

```python
def load_model(self):
    # ...
    elif self.model_name == "your_model":
        self.model = models.your_model(weights=...)
```

Ensure identical implementation in both engines!
