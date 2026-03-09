"""TensorRT inference engine via ONNX.

This module implements TensorRT-accelerated inference by:
1. Exporting PyTorch model to ONNX
2. Building TensorRT engine from ONNX
3. Running optimized inference with TensorRT runtime

Caches built engines to avoid rebuilding. Falls back to PyTorch if TensorRT unavailable.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

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
    logger.info("TensorRT is available")
except ImportError as e:
    TENSORRT_AVAILABLE = False
    logger.warning(f"TensorRT not available: {e}. Will use PyTorch fallback.")


class TensorRTEngine(InferenceEngine):
    """
    TensorRT inference engine via ONNX.

    Workflow:
    1. Export PyTorch model to ONNX (cached)
    2. Build TensorRT engine from ONNX (cached per batch size)
    3. Run inference with TensorRT runtime

    Falls back to PyTorch CUDA if TensorRT unavailable.
    """

    def __init__(self, model_name: str, cache_dir: str = "./cache/tensorrt"):
        """
        Initialize TensorRT engine.

        Args:
            model_name: Model identifier
            cache_dir: Directory to cache ONNX models and TensorRT engines
        """
        super().__init__(model_name)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available (required for TensorRT)")

        self._device = "cuda"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # TensorRT components
        self.trt_logger = None
        self.runtime = None
        self.context = None
        self.engine = None

        # Engine cache per batch size
        self.engine_cache: Dict[int, Any] = {}

        # Fallback to PyTorch if TensorRT not available
        self.use_fallback = not TENSORRT_AVAILABLE
        if self.use_fallback:
            logger.warning(
                f"TensorRT not available for {model_name}. "
                "Using PyTorch CUDA fallback."
            )
            self.fallback_model = None

    def load_model(self) -> None:
        """
        Load model for TensorRT inference.

        If TensorRT available: Exports to ONNX (cached)
        If TensorRT unavailable: Uses PyTorch CUDA fallback
        """
        if self.use_fallback:
            self._load_fallback_model()
        else:
            self._export_to_onnx()

        self._loaded = True

    def _load_fallback_model(self) -> None:
        """Load PyTorch CUDA model as fallback."""
        logger.info(
            f"event=load_fallback engine=tensorrt model={self.model_name}"
        )

        # Load pretrained model (identical to TorchCUDAEngine)
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

        logger.info(f"event=fallback_loaded engine=tensorrt model={self.model_name}")

    def _export_to_onnx(self) -> None:
        """Export PyTorch model to ONNX format (cached)."""
        onnx_path = self.cache_dir / f"{self.model_name}.onnx"

        if onnx_path.exists():
            logger.info(
                f"event=onnx_cached path={onnx_path} model={self.model_name}"
            )
            return

        logger.info(f"event=export_onnx_start model={self.model_name}")

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

        # Export to ONNX
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
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        logger.info(
            f"event=onnx_exported path={onnx_path} size_mb={onnx_path.stat().st_size / 1024 / 1024:.2f}"
        )

    def _build_engine(self, batch_size: int) -> Any:
        """
        Build TensorRT engine from ONNX for specific batch size.

        Args:
            batch_size: Batch size to optimize for

        Returns:
            TensorRT engine
        """
        # Check cache
        if batch_size in self.engine_cache:
            logger.debug(f"event=engine_cache_hit batch_size={batch_size}")
            return self.engine_cache[batch_size]

        engine_path = (
            self.cache_dir / f"{self.model_name}_batch{batch_size}.trt"
        )

        # Load cached engine if exists
        if engine_path.exists():
            logger.info(
                f"event=engine_load_cached path={engine_path} batch_size={batch_size}"
            )
            with open(engine_path, "rb") as f:
                engine_data = f.read()

            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.trt_logger)
            engine = self.runtime.deserialize_cuda_engine(engine_data)

            self.engine_cache[batch_size] = engine
            return engine

        # Build new engine
        logger.info(
            f"event=engine_build_start model={self.model_name} batch_size={batch_size}"
        )

        onnx_path = self.cache_dir / f"{self.model_name}.onnx"

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
                for error in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Set optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            (1, 3, 224, 224),  # min
            (batch_size, 3, 224, 224),  # opt
            (batch_size, 3, 224, 224),  # max
        )
        config.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(
            f"event=engine_built path={engine_path} "
            f"size_mb={len(serialized_engine) / 1024 / 1024:.2f}"
        )

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
            Latency in seconds

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
        """Run inference using TensorRT."""
        batch_size = inputs.shape[0]

        # Build engine for this batch size
        engine = self._build_engine(batch_size)
        context = engine.create_execution_context()

        # Prepare inputs and outputs
        input_np = inputs.cpu().numpy().astype(np.float32)

        # Allocate device memory
        d_input = cuda.mem_alloc(input_np.nbytes)
        output_shape = (batch_size, 1000)  # ImageNet classes
        output_np = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output_np.nbytes)

        # Copy input to device
        cuda.memcpy_htod(d_input, input_np)

        # Set input shape
        context.set_input_shape("input", input_np.shape)

        # Synchronize before timing
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Execute inference
        context.execute_v2([int(d_input), int(d_output)])

        # Synchronize after inference
        torch.cuda.synchronize()
        end = time.perf_counter()

        # Copy output back (not timed)
        cuda.memcpy_dtoh(output_np, d_output)

        return end - start

    @property
    def name(self) -> str:
        """Return engine name."""
        if self.use_fallback:
            return "TensorRT (PyTorch Fallback)"
        return "TensorRT"

    @property
    def device(self) -> str:
        """Return device name."""
        return self._device

    def metadata(self) -> Dict[str, Any]:
        """Return TensorRT engine metadata."""
        meta = super().metadata()
        meta.update({
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "tensorrt_available": TENSORRT_AVAILABLE,
            "using_fallback": self.use_fallback,
            "cache_dir": str(self.cache_dir),
            "cached_engines": list(self.engine_cache.keys()),
        })

        if TENSORRT_AVAILABLE:
            meta["tensorrt_version"] = trt.__version__

        return meta
