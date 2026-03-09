"""Benchmark runner orchestration.

This module coordinates benchmark execution across different inference engines,
ensuring reproducible and statistically valid measurements.
"""
import asyncio
import logging
from typing import List

from ..engines.base import InferenceEngine
from ..engines.tensorrt import TensorRTEngine
from ..engines.torch_cpu import TorchCPUEngine
from ..engines.torch_cuda import TorchCUDAEngine
from ..schemas.bench import BatchStats, EngineType
from .methodology import calculate_batch_statistics, validate_benchmark_config
from .workloads import prepare_fixed_inputs

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark execution across different engines."""

    def __init__(self):
        self.engines: dict[EngineType, type[InferenceEngine]] = {
            EngineType.PYTORCH_CPU: TorchCPUEngine,
            EngineType.PYTORCH_GPU: TorchCUDAEngine,
            EngineType.TENSORRT: TensorRTEngine,
        }

    async def run_benchmark(
        self,
        model_name: str,
        engine_type: EngineType,
        batch_sizes: List[int],
        num_iterations: int,
        warmup_iterations: int,
    ) -> List[BatchStats]:
        """
        Run benchmark for specified configuration.

        This method orchestrates the complete benchmarking workflow:
        1. Validate configuration
        2. Load model into selected engine
        3. For each batch size:
           - Generate fixed inputs
           - Run warmup iterations (excluded from stats)
           - Run measured iterations
           - Calculate statistics (p50, p95, mean, stddev, throughput)

        Args:
            model_name: Model identifier (e.g., 'resnet50')
            engine_type: Inference engine to use
            batch_sizes: List of batch sizes to test
            num_iterations: Number of measured iterations per batch
            warmup_iterations: Number of warmup iterations (excluded from stats)

        Returns:
            List of BatchStats, one per batch size

        Raises:
            ValueError: If configuration is invalid or engine not supported
        """
        # Validate configuration
        validate_benchmark_config(batch_sizes, num_iterations, warmup_iterations)

        logger.info(
            f"event=runner_start model={model_name} engine={engine_type} "
            f"batches={batch_sizes} iterations={num_iterations} warmup={warmup_iterations}"
        )

        # Get engine class
        engine_class = self.engines.get(engine_type)
        if engine_class is None:
            raise ValueError(f"Unsupported engine type: {engine_type}")

        # Initialize engine
        engine = engine_class(model_name=model_name)
        await asyncio.to_thread(engine.load_model)

        results: List[BatchStats] = []

        for batch_size in batch_sizes:
            logger.info(f"event=batch_start batch_size={batch_size}")

            # Prepare fixed inputs
            inputs = prepare_fixed_inputs(batch_size, engine.input_shape)

            # Warmup phase (results discarded)
            logger.debug(f"event=warmup_start batch_size={batch_size} iterations={warmup_iterations}")
            for i in range(warmup_iterations):
                await asyncio.to_thread(engine.infer, inputs)
            logger.debug(f"event=warmup_complete batch_size={batch_size}")

            # Measurement phase (only these results are analyzed)
            logger.debug(f"event=measure_start batch_size={batch_size} iterations={num_iterations}")
            latencies: List[float] = []
            for i in range(num_iterations):
                latency = await asyncio.to_thread(engine.infer, inputs)
                latencies.append(latency)
                if (i + 1) % 20 == 0:
                    logger.debug(f"event=progress batch_size={batch_size} completed={i+1}/{num_iterations}")

            # Calculate statistics (warmup excluded)
            stats = calculate_batch_statistics(batch_size, latencies)
            results.append(stats)

            logger.info(
                f"event=batch_complete batch_size={batch_size} "
                f"p50_ms={stats.latency_p50_ms:.2f} "
                f"p95_ms={stats.latency_p95_ms:.2f} "
                f"throughput_rps={stats.throughput_req_per_sec:.2f}"
            )

        return results
