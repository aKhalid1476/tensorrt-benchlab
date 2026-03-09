# TensorRT Implementation Guide

## Overview

The TensorRT engine provides optimized inference through:
1. **ONNX Export**: PyTorch → ONNX (cached)
2. **Engine Building**: ONNX → TensorRT engine (cached per batch size)
3. **Optimized Inference**: TensorRT runtime with CUDA

## Architecture

```
┌──────────────┐
│ PyTorch Model│
└──────┬───────┘
       │ torch.onnx.export()
       ▼
┌──────────────┐
│  ONNX File   │ (cached: resnet50.onnx)
└──────┬───────┘
       │ TensorRT Builder
       ▼
┌──────────────┐
│ TensorRT     │ (cached per batch: resnet50_batch1.trt)
│ Engine       │
└──────┬───────┘
       │ TensorRT Runtime
       ▼
┌──────────────┐
│   Inference  │ (optimized CUDA kernels)
└──────────────┘
```

## Implementation Details

### 1. ONNX Export

**Location:** `_export_to_onnx()`

```python
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=14,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
```

**Features:**
- ✅ Dynamic batch size support
- ✅ Cached to avoid re-export
- ✅ Opset version 14 for compatibility

### 2. TensorRT Engine Building

**Location:** `_build_engine(batch_size)`

```python
builder = trt.Builder(logger)
network = builder.create_network(EXPLICIT_BATCH)
parser = trt.OnnxParser(network, logger)

# Parse ONNX
parser.parse(onnx_file)

# Configure optimization
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# Set optimization profile
profile = builder.create_optimization_profile()
profile.set_shape("input", min, opt, max)
config.add_optimization_profile(profile)

# Build
engine = builder.build_serialized_network(network, config)
```

**Features:**
- ✅ Per-batch-size optimization
- ✅ Engine caching (avoids rebuild)
- ✅ 1GB workspace memory
- ✅ Optimization profiles for dynamic shapes

### 3. Inference Runtime

**Location:** `_infer_tensorrt(inputs)`

```python
# Allocate device memory
d_input = cuda.mem_alloc(input_np.nbytes)
d_output = cuda.mem_alloc(output_np.nbytes)

# Copy input to device
cuda.memcpy_htod(d_input, input_np)

# CRITICAL: Synchronize before timing
torch.cuda.synchronize()
start = time.perf_counter()

# Execute inference
context.execute_v2([int(d_input), int(d_output)])

# CRITICAL: Synchronize after inference
torch.cuda.synchronize()
end = time.perf_counter()
```

**Key Points:**
- ✅ CUDA synchronization for accurate timing
- ✅ Direct device memory management
- ✅ Preprocessing excluded from timing

## Fallback Behavior

**When TensorRT unavailable:**
```python
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
```

**Fallback mode:**
- Uses PyTorch CUDA engine
- Identical behavior to `TorchCUDAEngine`
- Automatic and transparent
- Logged as warning

## Caching Strategy

### ONNX Files

**Location:** `./cache/tensorrt/{model_name}.onnx`

Example: `./cache/tensorrt/resnet50.onnx`

**Cached once per model**

### TensorRT Engines

**Location:** `./cache/tensorrt/{model_name}_batch{size}.trt`

Examples:
- `./cache/tensorrt/resnet50_batch1.trt`
- `./cache/tensorrt/resnet50_batch4.trt`
- `./cache/tensorrt/resnet50_batch8.trt`

**Cached per batch size for optimal performance**

## Installation

### Requirements

```bash
# CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# TensorRT
# Download from: https://developer.nvidia.com/tensorrt
# Or install via pip (if available):
pip install tensorrt

# PyCUDA
pip install pycuda
```

### Verification

```python
from app.engines.tensorrt import TENSORRT_AVAILABLE

if TENSORRT_AVAILABLE:
    print("TensorRT is ready\!")
else:
    print("TensorRT not available, will use PyTorch fallback")
```

## Usage

```python
from app.engines.tensorrt import TensorRTEngine
import torch

# Initialize
engine = TensorRTEngine(
    model_name="resnet50",
    cache_dir="./cache/tensorrt"  # Optional
)

# Load model (exports ONNX, builds engine on first run)
engine.load_model()
print(f"Engine: {engine.name}")
print(f"Using fallback: {engine.use_fallback}")

# Run inference
inputs = torch.randn(4, 3, 224, 224)  # batch_size=4
latency = engine.infer(inputs)
print(f"Latency: {latency*1000:.2f}ms")

# Check metadata
meta = engine.metadata()
print(f"Cached engines: {meta['cached_engines']}")
```

## Testing

```bash
cd backend

# Install test dependencies
pip install -e .[dev]

# Run TensorRT tests
pytest tests/test_tensorrt_engine.py -v

# Run specific test
pytest tests/test_tensorrt_engine.py::TestTensorRTEngine::test_infer_single_batch -v

# Skip if TensorRT unavailable
pytest tests/test_tensorrt_engine.py -v --tb=short
```

## Performance Comparison

**ResNet50, batch_size=1:**

| Engine | Latency (ms) | Speedup |
|--------|--------------|---------|
| PyTorch CPU | ~100ms | 1x |
| PyTorch CUDA | ~15ms | 6.7x |
| **TensorRT** | ~**5ms** | **20x** |

*Actual numbers vary by hardware*

## Troubleshooting

### TensorRT Import Error

```
ImportError: No module named 'tensorrt'
```

**Solution:** Install TensorRT and PyCUDA
```bash
pip install tensorrt pycuda
```

### ONNX Export Fails

```
RuntimeError: ONNX export failed
```

**Solution:** Check PyTorch version compatibility
```bash
pip install --upgrade torch torchvision onnx
```

### Engine Build Fails

```
RuntimeError: Failed to build TensorRT engine
```

**Solution:**
1. Check CUDA compatibility
2. Increase workspace memory
3. Verify ONNX model validity

### Fallback Always Used

**Check:** `engine.use_fallback`

**Possible causes:**
- TensorRT not installed
- Import error during initialization
- CUDA version mismatch

## Best Practices

1. **Cache Management**
   - Keep cache directory for faster restarts
   - Clear cache when changing models
   - Monitor cache size (~100MB per model)

2. **Batch Size Optimization**
   - Build engines for commonly used batch sizes
   - Avoid building too many engines
   - Use dynamic shapes when appropriate

3. **Memory Management**
   - Adjust workspace size based on GPU memory
   - Monitor GPU memory usage
   - Use smaller batch sizes if OOM errors

4. **Performance Tuning**
   - Benchmark different precision modes (FP32/FP16)
   - Test various workspace sizes
   - Profile with Nsight Systems

## Future Enhancements

- [ ] FP16 precision support
- [ ] INT8 quantization
- [ ] Multi-GPU support
- [ ] Dynamic shape optimization
- [ ] Custom plugins for unsupported ops
