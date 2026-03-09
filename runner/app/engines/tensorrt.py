"""Production TensorRT inference engine via ONNX.

This module implements production-grade TensorRT-accelerated inference:
1. Export PyTorch model to ONNX with dynamic batch dimension
2. Build TensorRT engine with optimization profiles
3. Cache engines on disk (keyed by model/precision/batch/shape)
4. Run optimized inference with proper CUDA synchronization
5. Detailed timing breakdown (preprocessing/forward/postprocessing)

Falls back to PyTorch CUDA if TensorRT unavailable.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.models as models

from .base import InferenceEngine

logger = logging.getLogger(__name__)

# Check TensorRT availability
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # Initializes CUDA context

    TENSORRT_AVAILABLE = True
    logger.info(f"event=tensorrt_available version={trt.__version__}")
except ImportError as e:
    TENSORRT_AVAILABLE = False
    logger.warning(f"event=tensorrt_unavailable error={e}")


class TensorRTEngine(InferenceEngine):
    """
    Production TensorRT inference engine via ONNX.

    Features:
    - Real ONNX export with dynamic batch dimension
    - TensorRT engine building with optimization profiles
    - Disk caching (keyed by model/precision/max_batch/input_shape)
    - FP16 precision support (with FP32 fallback)
    - Detailed timing breakdown
    - Proper CUDA synchronization
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "./cache/tensorrt",
        precision: str = "fp16",
    ):
        """
        Initialize TensorRT engine.

        Args:
            model_name: Model identifier (e.g., 'resnet50')
            cache_dir: Directory to cache ONNX models and TensorRT engines
            precision: Precision mode ('fp16' or 'fp32')
        """
        super().__init__(model_name)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available (required for TensorRT)")

        self._device = "cuda"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.precision = precision

        # TensorRT components
        self.trt_logger = None
        self.runtime = None
        self.engine_cache: Dict[int, Any] = {}

        # Fallback to PyTorch if TensorRT not available
        self.use_fallback = not TENSORRT_AVAILABLE
        if self.use_fallback:
            logger.warning(
                f"event=fallback_mode model={model_name} reason='TensorRT unavailable'"
            )
            self.fallback_model = None
            self.precision = "fp32"  # Fallback always uses FP32

    def load_model(self) -> None:
        """Load model for TensorRT inference or fallback."""
        if self.use_fallback:
            self._load_fallback_model()
        else:
            # Just export ONNX; engines built per batch size on demand
            self._export_to_onnx()

        self._loaded = True

    def _load_fallback_model(self) -> None:
        """Load PyTorch CUDA model as fallback."""
        logger.info(f"event=load_fallback model={self.model_name}")

        if self.model_name == "resnet50":
            self.fallback_model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT
            )
        elif self.model_name == "mobilenet_v2":
            self.fallback_model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.fallback_model.eval()
        self.fallback_model.to(self._device)
        logger.info(f"event=fallback_loaded model={self.model_name}")

    def _export_to_onnx(self) -> None:
        """Export PyTorch model to ONNX format with dynamic batch dimension."""
        onnx_path = self.cache_dir / f"{self.model_name}.onnx"

        if onnx_path.exists():
            logger.info(f"event=onnx_cached path={onnx_path}")
            return

        logger.info(f"event=onnx_export_start model={self.model_name}")

        # Load PyTorch model
        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model.eval()
        model.to(self._device)

        # Dummy input for ONNX export
        dummy_input = torch.randn(1, 3, 224, 224, device=self._device)

        # Export to ONNX with dynamic batch dimension
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},  # Dynamic batch dimension
                "output": {0: "batch_size"},
            },
        )

        size_mb = onnx_path.stat().st_size / 1024 / 1024
        logger.info(f"event=onnx_exported path={onnx_path} size_mb={size_mb:.2f}")

    def _get_engine_cache_key(self, batch_size: int) -> str:
        """
        Generate cache key for TensorRT engine.

        Key format: {model}_{precision}_batch{max_batch}_{input_shape_hash}.trt
        """
        input_shape_str = f"{self.input_shape[0]}x{self.input_shape[1]}x{self.input_shape[2]}"
        key = f"{self.model_name}_{self.precision}_batch{batch_size}_{input_shape_str}.trt"
        return key

    def _save_engine_metadata(self, engine_path: Path, batch_size: int) -> None:
        """Save engine metadata alongside the engine file."""
        metadata = {
            "model_name": self.model_name,
            "precision": self.precision,
            "max_batch_size": batch_size,
            "input_shape": list(self.input_shape),
            "tensorrt_version": trt.__version__ if TENSORRT_AVAILABLE else None,
            "created_at": time.time(),
        }

        metadata_path = engine_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _build_engine(self, batch_size: int) -> Any:
        """
        Build TensorRT engine from ONNX for specific batch size.

        Uses optimization profiles for the target batch size.
        Supports FP16 precision if available.

        Args:
            batch_size: Maximum batch size to optimize for

        Returns:
            TensorRT engine
        """
        # Check cache
        if batch_size in self.engine_cache:
            logger.debug(f"event=engine_cache_hit batch_size={batch_size}")
            return self.engine_cache[batch_size]

        engine_cache_key = self._get_engine_cache_key(batch_size)
        engine_path = self.cache_dir / engine_cache_key

        # Load cached engine if exists
        if engine_path.exists():
            logger.info(f"event=engine_load_cached path={engine_path}")
            with open(engine_path, "rb") as f:
                engine_data = f.read()

            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.trt_logger)
            engine = self.runtime.deserialize_cuda_engine(engine_data)

            self.engine_cache[batch_size] = engine
            return engine

        # Build new engine
        logger.info(
            f"event=engine_build_start model={self.model_name} "
            f"batch_size={batch_size} precision={self.precision}"
        )

        onnx_path = self.cache_dir / f"{self.model_name}.onnx"
        if not onnx_path.exists():
            raise RuntimeError(f"ONNX file not found: {onnx_path}")

        # Create builder and network
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.trt_logger)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(str(parser.get_error(i)))
                raise RuntimeError(f"ONNX parse failed: {'; '.join(errors)}")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Enable FP16 if requested and supported
        if self.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("event=fp16_enabled")
        elif self.precision == "fp16":
            logger.warning(
                "event=fp16_not_supported fallback=fp32 reason='platform does not support FP16'"
            )
            self.precision = "fp32"

        # Set optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        input_shape = self.input_shape

        # Define min, opt, max shapes for dynamic batch
        profile.set_shape(
            "input",
            (1, *input_shape),  # min: batch=1
            (batch_size, *input_shape),  # opt: target batch size
            (batch_size, *input_shape),  # max: target batch size
        )
        config.add_optimization_profile(profile)

        # Build engine
        logger.info("event=engine_building (this may take a few minutes)")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        size_mb = len(serialized_engine) / 1024 / 1024
        logger.info(f"event=engine_built path={engine_path} size_mb={size_mb:.2f}")

        # Save metadata
        self._save_engine_metadata(engine_path, batch_size)

        # Deserialize for use
        self.runtime = trt.Runtime(self.trt_logger)
        engine = self.runtime.deserialize_cuda_engine(serialized_engine)

        self.engine_cache[batch_size] = engine
        return engine

    def infer(self, inputs: torch.Tensor) -> float:
        """
        Run inference with TensorRT or PyTorch fallback.

        Args:
            inputs: Input tensor with shape (batch_size, 3, 224, 224)

        Returns:
            Latency in seconds (forward pass only, excludes preprocessing)

        Raises:
            RuntimeError: If model not loaded
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.use_fallback:
            return self._infer_fallback(inputs)
        else:
            return self._infer_tensorrt(inputs)

    def _infer_fallback(self, inputs: torch.Tensor) -> float:
        """Run inference using PyTorch CUDA fallback."""
        inputs = inputs.to(self._device)

        # Synchronize before timing
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = self.fallback_model(inputs)

        # Synchronize after inference
        torch.cuda.synchronize()
        end = time.perf_counter()

        return end - start

    def _infer_tensorrt(self, inputs: torch.Tensor) -> float:
        """
        Run inference using TensorRT with proper CUDA synchronization.

        Returns forward pass latency only (preprocessing excluded).
        """
        batch_size = inputs.shape[0]

        # Build engine for this batch size
        engine = self._build_engine(batch_size)
        context = engine.create_execution_context()

        # Prepare inputs (preprocessing - not timed)
        input_np = inputs.cpu().numpy().astype(np.float32)

        # Allocate device memory
        d_input = cuda.mem_alloc(input_np.nbytes)
        output_shape = (batch_size, 1000)  # ImageNet classes
        output_np = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output_np.nbytes)

        # Copy input to device (preprocessing - not timed)
        cuda.memcpy_htod(d_input, input_np)

        # Set input shape
        context.set_input_shape("input", input_np.shape)

        # CRITICAL: Synchronize before timing
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Execute inference (forward pass - TIMED)
        context.execute_v2([int(d_input), int(d_output)])

        # CRITICAL: Synchronize after inference
        torch.cuda.synchronize()
        end = time.perf_counter()

        # Copy output back (postprocessing - not timed)
        cuda.memcpy_dtoh(output_np, d_output)

        return end - start

    def infer_with_breakdown(
        self, inputs: torch.Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """
        Run inference and return detailed timing breakdown.

        Returns:
            Tuple of (total_latency, breakdown_dict)
            breakdown_dict contains: preprocessing_ms, forward_ms, postprocessing_ms
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.use_fallback:
            # Fallback doesn't have detailed breakdown
            latency = self._infer_fallback(inputs)
            return latency, {
                "preprocessing_ms": 0.0,
                "forward_ms": latency * 1000,
                "postprocessing_ms": 0.0,
            }

        # TensorRT with detailed timing
        batch_size = inputs.shape[0]
        engine = self._build_engine(batch_size)
        context = engine.create_execution_context()

        # Preprocessing timing
        torch.cuda.synchronize()
        prep_start = time.perf_counter()

        input_np = inputs.cpu().numpy().astype(np.float32)
        d_input = cuda.mem_alloc(input_np.nbytes)
        output_shape = (batch_size, 1000)
        output_np = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output_np.nbytes)
        cuda.memcpy_htod(d_input, input_np)
        context.set_input_shape("input", input_np.shape)

        torch.cuda.synchronize()
        prep_end = time.perf_counter()
        preprocessing_ms = (prep_end - prep_start) * 1000

        # Forward pass timing
        torch.cuda.synchronize()
        forward_start = time.perf_counter()

        context.execute_v2([int(d_input), int(d_output)])

        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        forward_ms = (forward_end - forward_start) * 1000

        # Postprocessing timing
        torch.cuda.synchronize()
        post_start = time.perf_counter()

        cuda.memcpy_dtoh(output_np, d_output)

        torch.cuda.synchronize()
        post_end = time.perf_counter()
        postprocessing_ms = (post_end - post_start) * 1000

        total_latency = (forward_end - forward_start)  # Forward only for compatibility

        breakdown = {
            "preprocessing_ms": preprocessing_ms,
            "forward_ms": forward_ms,
            "postprocessing_ms": postprocessing_ms,
        }

        return total_latency, breakdown

    @property
    def name(self) -> str:
        """Return engine name."""
        if self.use_fallback:
            return "TensorRT (PyTorch Fallback)"
        return f"TensorRT ({self.precision.upper()})"

    @property
    def device(self) -> str:
        """Return device name."""
        return self._device

    def metadata(self) -> Dict[str, Any]:
        """Return TensorRT engine metadata."""
        meta = super().metadata()
        meta.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "tensorrt_available": TENSORRT_AVAILABLE,
                "using_fallback": self.use_fallback,
                "precision": self.precision,
                "cache_dir": str(self.cache_dir),
                "cached_engines": list(self.engine_cache.keys()),
            }
        )

        if TENSORRT_AVAILABLE:
            meta["tensorrt_version"] = trt.__version__

        return meta
