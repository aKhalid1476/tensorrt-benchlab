"""Workload generation for benchmarks."""
import numpy as np
import torch

# Fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def prepare_fixed_inputs(batch_size: int, input_shape: tuple[int, ...]) -> torch.Tensor:
    """
    Prepare fixed random inputs for benchmarking.

    Args:
        batch_size: Batch size
        input_shape: Input shape (C, H, W) for images

    Returns:
        Fixed random tensor inputs
    """
    # Create fixed random inputs (simulating image data)
    shape = (batch_size,) + input_shape
    inputs = torch.randn(shape, dtype=torch.float32)

    return inputs
