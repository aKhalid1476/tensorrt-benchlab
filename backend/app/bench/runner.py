"""Benchmark runner orchestration."""
import asyncio
import logging
from typing import List

from ..engines.base import InferenceEngine
from ..engines.tensorrt import TensorRTEngine
from ..engines.torch_cpu import TorchCPUEngine
from ..engines.torch_cuda import TorchCUDAEngine
from ..schemas.bench import BatchStats, EngineType
from .methodology import calculate_batch_statistics
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
        """Run benchmark for specified configuration."""
        logger.info(
            f"event=runner_start model={model_name} engine={engine_type} "
            f"batches={batch_sizes} iterations={num_iterations}"
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

            # Warmup
            logger.debug(f"event=warmup_start batch_size={batch_size} iterations={warmup_iterations}")
            for _ in range(warmup_iterations):
                await asyncio.to_thread(engine.infer, inputs)
            logger.debug(f"event=warmup_complete batch_size={batch_size}")

            # Measure
            logger.debug(f"event=measure_start batch_size={batch_size} iterations={num_iterations}")
            latencies = []
            for i in range(num_iterations):
                latency = await asyncio.to_thread(engine.infer, inputs)
                latencies.append(latency)
                if (i + 1) % 20 == 0:
                    logger.debug(f"event=progress batch_size={batch_size} completed={i+1}/{num_iterations}")

            # Calculate statistics
            stats = calculate_batch_statistics(batch_size, latencies)
            results.append(stats)

            logger.info(
                f"event=batch_complete batch_size={batch_size} "
                f"p50_ms={stats.latency_p50_ms:.2f} "
                f"p95_ms={stats.latency_p95_ms:.2f} "
                f"throughput_rps={stats.throughput_req_per_sec:.2f}"
            )

        return results
