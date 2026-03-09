# Engine Design

## Abstract Interface

All engines implement the `InferenceEngine` abstract base class:

```python
class InferenceEngine(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """Load model and prepare for inference."""

    @abstractmethod
    def infer(self, inputs: torch.Tensor) -> float:
        """Run inference and return latency in seconds."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable engine name."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Return device name."""

    def metadata(self) -> Dict[str, Any]:
        """Return engine metadata."""
```

## Implementations

### 1. PyTorch CPU Engine (`torch_cpu.py`)

**Features:**
- Uses torchvision pretrained models
- Runs on CPU
- `eval()` mode for consistent behavior
- `torch.no_grad()` to disable gradients
- Precise timing with `time.perf_counter()`

**Supported Models:**
- ResNet50 (1000 classes)
- MobileNetV2 (1000 classes)

**Usage:**
```python
engine = TorchCPUEngine(model_name="resnet50")
engine.load_model()

inputs = torch.randn(4, 3, 224, 224)  # batch_size=4
latency = engine.infer(inputs)  # seconds
```

### 2. PyTorch CUDA Engine (`torch_cuda.py`)

**Features:**
- Identical model weights as CPU engine
- Runs on NVIDIA GPU with CUDA
- `eval()` mode for consistent behavior
- `torch.no_grad()` to disable gradients
- **CRITICAL**: Proper CUDA synchronization for accurate timing

**CUDA Synchronization:**
```python
def infer(self, inputs: torch.Tensor) -> float:
    inputs = inputs.to("cuda")

    # Synchronize BEFORE timing
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output = self.model(inputs)

    # Synchronize AFTER inference
    torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start
```

**Why synchronization matters:**
- CUDA operations are **asynchronous**
- Without `torch.cuda.synchronize()`, timing would be incorrect
- We measure only GPU compute time, not data transfer

### 3. TensorRT Engine (`tensorrt.py`)

**Status:** Placeholder (uses PyTorch as fallback)

**TODO:**
- ONNX export from PyTorch
- TensorRT engine compilation
- TensorRT inference runtime

## Preprocessing Consistency

**CRITICAL**: All engines use **identical preprocessing**:

1. **Input Shape**: `(batch_size, 3, 224, 224)`
   - 3 channels (RGB)
   - 224x224 resolution (ImageNet standard)

2. **No Preprocessing Applied**: Raw random tensors
   - In production, would apply ImageNet normalization:
     - mean = [0.485, 0.456, 0.406]
     - std = [0.229, 0.224, 0.225]

3. **Model Configuration**:
   - `model.eval()` mode
   - `torch.no_grad()` context
   - Pretrained ImageNet weights

## Testing

### Unit Tests (`test_engines.py`)

**Coverage:**
- ✅ Engine initialization
- ✅ Model loading
- ✅ Inference with various batch sizes
- ✅ Error handling (model not loaded, unsupported model)
- ✅ `eval()` mode verification
- ✅ `no_grad()` context verification
- ✅ Output shape validation
- ✅ Deterministic inference
- ✅ Metadata extraction
- ✅ CUDA synchronization (for GPU)
- ✅ CPU/CUDA consistency

**Run tests:**
```bash
cd backend
pytest tests/test_engines.py -v
```

### Consistency Tests

**Same Weights:**
```python
def test_same_model_weights():
    cpu_engine = TorchCPUEngine("resnet50")
    cuda_engine = TorchCUDAEngine("resnet50")

    cpu_engine.load_model()
    cuda_engine.load_model()

    # Weights should be identical
    cpu_weights = cpu_engine.model.conv1.weight.data
    cuda_weights = cuda_engine.model.conv1.weight.data.cpu()

    assert torch.allclose(cpu_weights, cuda_weights)
```

**Same Outputs:**
```python
def test_consistent_outputs():
    # Same inputs on both engines
    inputs = torch.randn(2, 3, 224, 224)

    cpu_output = cpu_engine.model(inputs)
    cuda_output = cuda_engine.model(inputs.cuda()).cpu()

    # Outputs should be very close
    assert torch.allclose(cpu_output, cuda_output, rtol=1e-3)
```

## Best Practices

### 1. Always Use eval() Mode
```python
model.eval()  # Disables dropout, batch norm updates
```

### 2. Always Use no_grad()
```python
with torch.no_grad():  # Disables gradient computation
    output = model(inputs)
```

### 3. CUDA Synchronization
```python
# BEFORE timing
torch.cuda.synchronize()

# Run inference
output = model(inputs)

# AFTER inference
torch.cuda.synchronize()
```

### 4. Error Handling
```python
if not self._loaded:
    raise RuntimeError("Model not loaded")
```

## Adding New Models

1. **Add to engine implementations:**
```python
def load_model(self):
    if self.model_name == "your_model":
        self.model = models.your_model(weights=...)
    # ...
```

2. **Update both CPU and CUDA engines** identically

3. **Add tests** for new model

## Performance Notes

**ResNet50:**
- CPU: ~50-100ms per batch (depends on CPU)
- GPU: ~5-15ms per batch (depends on GPU)

**MobileNetV2:**
- CPU: ~20-40ms per batch
- GPU: ~2-5ms per batch

These are approximate values for batch_size=1.
