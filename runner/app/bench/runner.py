"""Benchmark runner for executing benchmarks on GPU host."""
import asyncio
import logging
import time
from typing import List

from contracts import BatchStats, EngineType, TimingBreakdown

from ..engines.base import InferenceEngine
from ..engines.tensorrt import TensorRTEngine
from ..engines.torch_cpu import TorchCPUEngine
from ..engines.torch_cuda import TorchCUDAEngine
from .methodology import calculate_batch_statistics, validate_benchmark_config
from .workloads import prepare_fixed_inputs

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Executes benchmarks across different inference engines."""

    def __init__(self):
        """Initialize runner."""
        self.engines = {
            EngineType.PYTORCH_CPU: TorchCPUEngine,
            EngineType.PYTORCH_CUDA: TorchCUDAEngine,
            EngineType.TENSORRT: TensorRTEngine,
        }

    async def run_benchmark(
        self,
        model_name: str,
        engine_type: EngineType,
        batch_sizes: List[int],
        num_iterations: int,
        warmup_iterations: int,
    ) -> tuple[List[BatchStats], TimingBreakdown]:
        """
        Run benchmark for specified configuration.

        Args:
            model_name: Model identifier
            engine_type: Inference engine to use
            batch_sizes: List of batch sizes to test
            num_iterations: Number of measured iterations per batch
            warmup_iterations: Number of warmup iterations

        Returns:
            Tuple of (batch_stats, timing_breakdown)
        """
        start_time = time.time()

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
        engine: InferenceEngine = engine_class(model_name=model_name)

        # Load model
        load_start = time.time()
        await asyncio.to_thread(engine.load_model)
        load_duration = time.time() - load_start

        logger.info(
            f"event=model_loaded engine={engine.name} device={engine.device} "
            f"load_time_sec={load_duration:.2f}"
        )

        # Run benchmarks for each batch size
        results: List[BatchStats] = []
        warmup_start = time.time()
        measurement_total = 0.0

        for batch_size in batch_sizes:
            logger.info(f"event=batch_start batch_size={batch_size}")

            # Prepare fixed inputs
            inputs = prepare_fixed_inputs(batch_size, engine.input_shape)

            # Warmup phase
            logger.debug(
                f"event=warmup_start batch_size={batch_size} iterations={warmup_iterations}"
            )
            for i in range(warmup_iterations):
                await asyncio.to_thread(engine.infer, inputs)

            warmup_duration = time.time() - warmup_start

            # Measurement phase
            measure_start = time.time()
            logger.debug(
                f"event=measure_start batch_size={batch_size} iterations={num_iterations}"
            )
            latencies: List[float] = []
            for i in range(num_iterations):
                latency = await asyncio.to_thread(engine.infer, inputs)
                latencies.append(latency)

            measurement_duration = time.time() - measure_start
            measurement_total += measurement_duration

            # Calculate statistics
            stats = calculate_batch_statistics(batch_size, latencies, engine.name)
            results.append(stats)

            logger.info(
                f"event=batch_complete batch_size={batch_size} "
                f"p50_ms={stats.latency_p50_ms:.2f} "
                f"p95_ms={stats.latency_p95_ms:.2f} "
                f"throughput_rps={stats.throughput_req_per_sec:.2f}"
            )

        total_duration = time.time() - start_time

        timing = TimingBreakdown(
            total_duration_sec=total_duration,
            model_load_sec=load_duration,
            warmup_duration_sec=warmup_duration,
            measurement_duration_sec=measurement_total,
        )

        return results, timing
