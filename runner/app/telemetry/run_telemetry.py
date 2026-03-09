"""Run-scoped telemetry collection with relative timestamps.

This module provides telemetry sampling that's correlated to specific benchmark runs.
Timestamps are relative to the run start (t_ms) for easier analysis.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import List, Optional

from contracts import TelemetrySample

from .nvml_sampler import NVMLSampler

logger = logging.getLogger(__name__)


class RunTelemetry:
    """
    Run-scoped telemetry collector.

    Collects GPU metrics during a benchmark run with timestamps relative to run start.
    """

    def __init__(self, sampler: NVMLSampler, run_id: str):
        """
        Initialize run telemetry.

        Args:
            sampler: Global NVML sampler instance
            run_id: Unique run identifier
        """
        self.sampler = sampler
        self.run_id = run_id
        self.start_time: Optional[float] = None
        self.samples: List[TelemetrySample] = []
        self._sampling_task: Optional[asyncio.Task] = None
        self._should_stop = False

    async def start(self) -> None:
        """Start collecting telemetry for this run."""
        if not self.sampler.enabled:
            logger.warning(
                f"event=telemetry_disabled run_id={self.run_id} reason='NVML unavailable'"
            )
            return

        self.start_time = time.time()
        self._should_stop = False
        self._sampling_task = asyncio.create_task(self._collect_loop())
        logger.info(f"event=telemetry_start run_id={self.run_id}")

    async def stop(self) -> None:
        """Stop collecting telemetry."""
        if self._sampling_task is None:
            return

        self._should_stop = True
        try:
            await asyncio.wait_for(self._sampling_task, timeout=1.0)
        except asyncio.TimeoutError:
            self._sampling_task.cancel()
        except asyncio.CancelledError:
            pass

        logger.info(
            f"event=telemetry_stop run_id={self.run_id} samples_collected={len(self.samples)}"
        )

    async def _collect_loop(self) -> None:
        """Continuously collect telemetry samples."""
        while not self._should_stop:
            try:
                # Take sample from global sampler
                sample = self.sampler.sample()

                # Convert to relative timestamp
                if self.start_time is not None:
                    t_ms = (time.time() - self.start_time) * 1000.0
                else:
                    t_ms = 0.0

                # Create run-scoped sample with relative timestamp
                run_sample = TelemetrySample(
                    t_ms=t_ms,
                    gpu_utilization_percent=sample.gpu_utilization_percent,
                    memory_used_mb=sample.memory_used_mb,
                    memory_total_mb=sample.memory_total_mb,
                    temperature_celsius=sample.temperature_celsius,
                    power_usage_watts=sample.power_usage_watts,
                )

                self.samples.append(run_sample)

                # Sample every 200ms
                await asyncio.sleep(0.2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"event=telemetry_error run_id={self.run_id} error={e}",
                    exc_info=True,
                )
                await asyncio.sleep(0.2)

    def get_samples(self) -> List[TelemetrySample]:
        """Get all samples collected for this run."""
        return self.samples.copy()
