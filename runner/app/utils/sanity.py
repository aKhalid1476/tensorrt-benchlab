"""Cross-engine sanity checks for correctness validation.

This module compares outputs across different engines (CPU, CUDA, TensorRT)
to ensure they produce consistent results for the same input.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..engines.base import InferenceEngine

logger = logging.getLogger(__name__)


def run_sanity_check(
    engines: Dict[str, InferenceEngine],
    batch_size: int = 1,
    tolerance: float = 1e-3,
) -> Tuple[bool, Dict[str, any]]:
    """
    Run cross-engine sanity check.

    Compares top-1 predicted class across all engines for a fixed input.

    Args:
        engines: Dict mapping engine names to loaded InferenceEngine instances
        batch_size: Batch size for test input
        tolerance: Tolerance for output comparison

    Returns:
        Tuple of (passed, details_dict)
        details_dict contains: top1_classes, max_diff, matched
    """
    if len(engines) < 2:
        logger.warning("event=sanity_skip reason='need at least 2 engines'")
        return True, {"skipped": True}

    logger.info(f"event=sanity_check_start engines={list(engines.keys())}")

    # Create fixed input (deterministic)
    torch.manual_seed(42)
    test_input = torch.randn(batch_size, 3, 224, 224)

    # Run inference on all engines and collect outputs
    outputs = {}
    top1_classes = {}

    for name, engine in engines.items():
        try:
            # Get raw output (we need the actual predictions, not just latency)
            output = _get_engine_output(engine, test_input)
            outputs[name] = output

            # Get top-1 class for each sample in batch
            top1 = torch.argmax(output, dim=1).cpu().numpy()
            top1_classes[name] = top1.tolist()

            logger.debug(
                f"event=sanity_engine_output engine={name} "
                f"top1_sample0={top1[0]} output_shape={output.shape}"
            )

        except Exception as e:
            logger.error(
                f"event=sanity_engine_failed engine={name} error={str(e)}",
                exc_info=True,
            )
            return False, {"error": str(e), "failed_engine": name}

    # Compare top-1 classes across engines
    reference_name = list(engines.keys())[0]
    reference_top1 = top1_classes[reference_name]

    matched = True
    mismatches = []

    for name, top1 in top1_classes.items():
        if name == reference_name:
            continue

        if top1 != reference_top1:
            matched = False
            mismatches.append(
                {
                    "engine": name,
                    "reference": reference_name,
                    "reference_top1": reference_top1,
                    "actual_top1": top1,
                }
            )

    # Calculate maximum difference in raw outputs
    max_diff = 0.0
    if len(outputs) >= 2:
        output_arrays = [outputs[name].cpu().numpy() for name in outputs.keys()]
        for i in range(len(output_arrays)):
            for j in range(i + 1, len(output_arrays)):
                diff = np.abs(output_arrays[i] - output_arrays[j]).max()
                max_diff = max(max_diff, diff)

    if matched:
        logger.info(
            f"event=sanity_check_passed engines={list(engines.keys())} "
            f"top1={reference_top1} max_diff={max_diff:.6f}"
        )
    else:
        logger.warning(
            f"event=sanity_check_failed mismatches={len(mismatches)} "
            f"max_diff={max_diff:.6f} details={mismatches}"
        )

    details = {
        "passed": matched,
        "top1_classes": top1_classes,
        "max_diff": float(max_diff),
        "matched": matched,
        "mismatches": mismatches if not matched else [],
    }

    return matched, details


def _get_engine_output(engine: InferenceEngine, inputs: torch.Tensor) -> torch.Tensor:
    """
    Get actual model output from engine (not just latency).

    This is a helper to extract predictions for sanity checking.
    """
    # We need to run inference and get the actual output tensor
    # The current engine interface only returns latency, so we need to
    # temporarily modify the inference to return outputs

    if hasattr(engine, "fallback_model") and engine.fallback_model is not None:
        # PyTorch fallback
        inputs = inputs.to(engine.device)
        with torch.no_grad():
            return engine.fallback_model(inputs)

    elif hasattr(engine, "model") and engine.model is not None:
        # Regular PyTorch engine
        inputs = inputs.to(engine.device)
        with torch.no_grad():
            return engine.model(inputs)

    elif hasattr(engine, "_infer_tensorrt"):
        # TensorRT engine - need to extract output
        return _get_tensorrt_output(engine, inputs)

    else:
        raise RuntimeError(f"Cannot extract output from engine: {engine.name}")


def _get_tensorrt_output(engine, inputs: torch.Tensor) -> torch.Tensor:
    """Extract actual output from TensorRT engine."""
    try:
        import pycuda.driver as cuda
    except ImportError:
        raise RuntimeError("PyCUDA required for TensorRT sanity check")

    batch_size = inputs.shape[0]

    # Build engine
    trt_engine = engine._build_engine(batch_size)
    context = trt_engine.create_execution_context()

    # Prepare inputs
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

    # Execute
    context.execute_v2([int(d_input), int(d_output)])

    # Copy output back
    cuda.memcpy_dtoh(output_np, d_output)

    # Convert to torch tensor
    return torch.from_numpy(output_np)
