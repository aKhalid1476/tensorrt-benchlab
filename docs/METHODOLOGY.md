# Benchmark Methodology

This document explains the rigorous methodology used in TensorRT BenchLab to ensure reproducible, trustworthy performance measurements.

## Overview

The goal is to measure **inference latency** and **throughput** for deep learning models across different execution engines (PyTorch CPU, PyTorch CUDA, TensorRT) with **statistical rigor** and **reproducibility**.

## Core Principles

### 1. Warmup Phase (Excluded from Measurements)

**Why warmup matters:**
- First few inferences include one-time initialization costs:
  - GPU kernel compilation (JIT)
  - CUDA context creation
  - Memory allocation
  - Cache warming
- These costs are not representative of steady-state performance

**Implementation:**
- Default: 10 warmup iterations
- Configurable via `warmup_iterations` parameter
- Warmup uses the **same batch sizes** as measurement phase
- Warmup iterations are **completely discarded** - not included in statistics

**Code:**
```python
# Warmup (not measured)
for _ in range(warmup_iterations):
    engine.infer(inputs)  # Discard results

# Measurement phase starts here
```

### 2. CUDA Synchronization for Accurate Timing

**The Problem:**
- CUDA operations are **asynchronous** by default
- `time.time()` before/after a CUDA call measures **kernel launch time**, not execution time
- This gives artificially fast measurements (microseconds instead of milliseconds)

**The Solution:**
```python
import torch
import time

# WRONG - asynchronous timing
start = time.time()
output = model(inputs)
end = time.time()  # ⚠️ Measures launch time, not execution

# CORRECT - synchronous timing
torch.cuda.synchronize()  # Wait for all previous ops
start = time.time()
output = model(inputs)
torch.cuda.synchronize()  # Wait for forward pass to complete
end = time.time()  # ✅ Measures actual execution time
```

**In TensorRT BenchLab:**
- All CUDA engines (`torch_cuda`, `tensorrt`) use `torch.cuda.synchronize()` before and after the forward pass
- CPU engine doesn't need synchronization (already synchronous)

### 3. Timing Breakdown

Each inference is timed with **three separate phases**:

1. **Preprocessing**: Image normalization, tensor preparation
2. **Forward Pass**: Model execution (synchronized)
3. **Postprocessing**: Argmax, top-k extraction

**Why separate timings:**
- Preprocessing/postprocessing costs are **constant** across engines
- Forward pass is what we're actually benchmarking
- Allows apples-to-apples comparison

**Implementation:**
```python
def infer_with_breakdown(self, inputs):
    # Preprocessing
    t_pre_start = time.time()
    preprocessed = self.preprocess(inputs)
    t_pre_end = time.time()

    # Forward pass (synchronized)
    torch.cuda.synchronize()
    t_fwd_start = time.time()
    outputs = self.model(preprocessed)
    torch.cuda.synchronize()
    t_fwd_end = time.time()

    # Postprocessing
    t_post_start = time.time()
    results = self.postprocess(outputs)
    t_post_end = time.time()

    return {
        "preprocessing_ms": (t_pre_end - t_pre_start) * 1000,
        "forward_ms": (t_fwd_end - t_fwd_start) * 1000,
        "postprocessing_ms": (t_post_end - t_post_start) * 1000,
    }
```

### 4. Statistical Measurements

For each (engine, batch_size) combination, we collect **N measurements** (default: 50 iterations).

**Statistics computed:**
- **p50 (median)**: Middle value - robust to outliers
- **p95**: 95th percentile - captures tail latency (important for user experience)
- **Mean**: Average latency
- **Stddev**: Standard deviation - measures variance
- **Throughput**: Images processed per second = `batch_size / (latency_sec)`

**Why percentiles matter:**
- Mean alone is misleading if there are outliers
- p50 (median) is robust to occasional spikes
- p95 shows worst-case user experience (95% of requests are faster than this)

**Code:**
```python
import numpy as np

latencies = [run_inference() for _ in range(num_iterations)]

stats = {
    "latency_ms_p50": np.percentile(latencies, 50),
    "latency_ms_p95": np.percentile(latencies, 95),
    "latency_ms_mean": np.mean(latencies),
    "latency_ms_stddev": np.std(latencies),
    "throughput_img_per_sec": batch_size / (np.mean(latencies) / 1000.0),
}
```

### 5. Reproducibility

**Fixed Random Seed:**
```python
torch.manual_seed(42)
np.random.seed(42)
```

**Fixed Input Set:**
- Generate N random tensors once at the start
- Reuse the same tensors for all iterations
- Prevents variance from different inputs

**Fixed Batch Sizes:**
- Default: [1, 4, 8, 16]
- Allows consistent comparison across runs

**Why this matters:**
- Same inputs → same outputs (deterministic)
- Different runs of the same configuration should produce identical results
- Enables A/B testing of optimization changes

### 6. Correctness Sanity Checks

**Cross-Engine Validation:**
- Run the **same input** through all engines
- Compare **top-1 predicted class** across engines
- If predictions differ → something is wrong (quantization error, incorrect export, etc.)

**Implementation:**
```python
def run_sanity_check(engines, batch_size=1):
    # Generate fixed input
    torch.manual_seed(42)
    inputs = torch.randn(batch_size, 3, 224, 224)

    predictions = {}
    for engine_name, engine in engines.items():
        output = engine.infer(inputs)
        top1_class = output.argmax(dim=1).item()
        predictions[engine_name] = top1_class

    # Check if all engines agree
    unique_preds = set(predictions.values())
    passed = len(unique_preds) == 1

    return passed, predictions
```

**Result:**
- Stored in `EnvironmentMetadata.sanity_check_passed`
- Warns if engines disagree
- Helps catch export/quantization issues early

## TensorRT-Specific Considerations

### 1. ONNX Export with Dynamic Batch Dimension

**Why dynamic batch:**
- TensorRT engines built for fixed batch sizes can't handle variable batches
- Dynamic batch axis allows single engine to handle [1, 4, 8, 16]

**Export:**
```python
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
```

### 2. TensorRT Engine Building with Optimization Profiles

**Optimization Profile:**
- TensorRT needs min/opt/max shapes for dynamic dimensions
- We set all three to the target batch size for optimal performance

```python
profile = builder.create_optimization_profile()
profile.set_shape(
    "input",
    (1, 3, 224, 224),           # min shape
    (batch_size, 3, 224, 224),  # optimal shape
    (batch_size, 3, 224, 224),  # max shape
)
config.add_optimization_profile(profile)
```

### 3. FP16 Precision

**Default: FP16 with FP32 fallback:**
```python
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    precision = "fp16"
else:
    precision = "fp32"
```

**Why FP16:**
- 2x faster inference on modern GPUs (Tensor Cores)
- 2x less memory usage
- Minimal accuracy loss for ResNet50

**Reported in metadata:**
- Precision is stored in `EnvironmentMetadata` for transparency

### 4. Disk Caching

**Cache Key:**
```
{model}_{precision}_batch{max_batch}_{input_shape}.trt
```

Example: `resnet50_fp16_batch16_3x224x224.trt`

**Why caching:**
- Engine building takes 30-60 seconds
- Cache lookup takes <1 second
- Enables fast repeated benchmarks

**Cache invalidation:**
- Delete `.trt` files to rebuild
- Different batch sizes use different engines

## Telemetry Methodology

### 1. Run-Scoped Sampling

**Problem:**
- Global telemetry sampler runs continuously
- Need to correlate GPU metrics to specific benchmark runs

**Solution:**
- Each run gets a unique `run_id`
- Start collecting at run start → stop at run end
- Store samples with **relative timestamps** (`t_ms` from run start)

### 2. Relative Timestamps

**Why relative timestamps:**
- Easier to correlate with benchmark phases
- First sample is at t=0, makes plotting simple
- Independent of system clock

**Implementation:**
```python
class RunTelemetry:
    async def start(self):
        self.start_time = time.time()

    async def _collect_loop(self):
        while not self._should_stop:
            sample = self.sampler.sample()
            t_ms = (time.time() - self.start_time) * 1000.0

            self.samples.append(TelemetrySample(
                t_ms=t_ms,  # Relative to run start
                gpu_utilization_percent=sample.gpu_utilization_percent,
                ...
            ))

            await asyncio.sleep(0.2)  # 200ms sampling rate
```

### 3. Sampling Rate

**200ms intervals:**
- Fast enough to capture spikes
- Not so fast that we overwhelm the system
- ~5 samples per second

**Collected metrics:**
- GPU utilization (%)
- Memory used / total (MB)
- Temperature (°C)
- Power usage (W)

## What's NOT Included in Timings

To ensure fair comparisons, the following are **excluded** from latency measurements:

**Excluded:**
- Model loading from disk
- ONNX export time
- TensorRT engine building time
- Warmup iterations
- Data loading from disk
- Result serialization

**Included (measured):**
- Preprocessing (normalization, reshaping)
- Forward pass (model execution)
- Postprocessing (argmax, top-k)

**Reported separately:**
- Model load time in `TimingBreakdown.model_load_sec`
- Warmup duration in `TimingBreakdown.warmup_duration_sec`
- Total run duration in `TimingBreakdown.total_duration_sec`

## Environment Metadata

Every benchmark run captures:

```python
{
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "gpu_driver_version": "525.60.13",
    "cuda_version": "12.1",
    "torch_version": "2.1.0",
    "tensorrt_version": "8.6.1",
    "python_version": "3.11.5",
    "cpu_model": "Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz",
    "timestamp": "2024-01-15T10:30:00Z",
    "git_commit": "a3f8d9c",
    "sanity_check_passed": true
}
```

**Why this matters:**
- Results are meaningless without environment context
- Different GPUs have different performance characteristics
- Driver/CUDA/TensorRT versions affect results
- Enables reproducibility and debugging

## Trustworthiness Checklist

✅ **Warmup iterations excluded**
✅ **CUDA operations synchronized**
✅ **Multiple iterations (N=50+) for statistics**
✅ **p50 and p95 reported (not just mean)**
✅ **Fixed random seed and inputs**
✅ **Cross-engine sanity checks**
✅ **Environment metadata captured**
✅ **Separate timing for preprocess/forward/postprocess**
✅ **TensorRT FP16 with fallback**
✅ **Disk caching for engine builds**
✅ **Run-scoped telemetry with relative timestamps**

## References

- [NVIDIA TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)
- [PyTorch Benchmark Utils](https://pytorch.org/docs/stable/benchmark_utils.html)
- [How to Benchmark PyTorch Models](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [CUDA Synchronization Pitfalls](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
