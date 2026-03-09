"""NVML GPU telemetry sampler with background sampling.

This module provides continuous GPU telemetry sampling using NVIDIA's NVML library.
Samples are taken every 200ms and stored in a ring buffer for retrieval.
"""
import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import List, Optional

from ..schemas.bench import TelemetrySample

logger = logging.getLogger(__name__)


class NVMLSampler:
    """
    GPU telemetry sampler using NVML.

    Continuously samples GPU metrics every 200ms in the background:
    - GPU utilization %
    - Memory used/total (MB)
    - Temperature (Celsius)
    - Power draw (Watts)

    Samples are stored in a ring buffer (default 1000 samples = ~200 seconds).
    """

    def __init__(self, max_samples: int = 1000, sample_interval_ms: int = 200):
        """
        Initialize NVML sampler.

        Args:
            max_samples: Maximum samples in ring buffer
            sample_interval_ms: Sampling interval in milliseconds
        """
        self.max_samples = max_samples
        self.sample_interval_ms = sample_interval_ms
        self.samples: deque[TelemetrySample] = deque(maxlen=max_samples)
        self.device_name = "Unknown"
        self._sampling_task: Optional[asyncio.Task] = None
        self._should_stop = False
        self._last_sample: Optional[TelemetrySample] = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.device_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.device_name, bytes):
                self.device_name = self.device_name.decode('utf-8')
            self.enabled = True
            logger.info(f"event=nvml_init device={self.device_name} max_samples={max_samples} interval_ms={sample_interval_ms}")
        except Exception as e:
            logger.warning(f"event=nvml_init_failed error={e}")
            self.enabled = False
            self.handle = None

    def sample(self) -> TelemetrySample:
        """
        Take a single telemetry sample.

        Returns:
            TelemetrySample with current GPU metrics
        """
        if not self.enabled:
            sample = TelemetrySample(
                timestamp=datetime.now(),
                gpu_utilization_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
            )
            self._last_sample = sample
            return sample

        try:
            import pynvml

            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)

            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = None

            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
            except:
                power = None

            sample = TelemetrySample(
                timestamp=datetime.now(),
                gpu_utilization_percent=float(util.gpu),
                memory_used_mb=float(mem_info.used / 1024 / 1024),
                memory_total_mb=float(mem_info.total / 1024 / 1024),
                temperature_celsius=float(temp) if temp is not None else None,
                power_usage_watts=float(power) if power is not None else None,
            )

            self.samples.append(sample)
            self._last_sample = sample
            return sample

        except Exception as e:
            logger.error(f"event=sample_failed error={e}")
            sample = TelemetrySample(
                timestamp=datetime.now(),
                gpu_utilization_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
            )
            self._last_sample = sample
            return sample

    async def _sampling_loop(self) -> None:
        """
        Background sampling loop.

        Continuously samples GPU metrics every sample_interval_ms until stopped.
        """
        logger.info(f"event=sampling_start interval_ms={self.sample_interval_ms}")

        while not self._should_stop:
            try:
                # Take sample (synchronous operation)
                self.sample()

                # Sleep for interval
                await asyncio.sleep(self.sample_interval_ms / 1000.0)

            except asyncio.CancelledError:
                logger.info("event=sampling_cancelled")
                break
            except Exception as e:
                logger.error(f"event=sampling_error error={e}", exc_info=True)
                # Continue sampling despite errors
                await asyncio.sleep(self.sample_interval_ms / 1000.0)

        logger.info("event=sampling_stopped")

    def start_sampling(self) -> None:
        """
        Start background sampling task.

        Creates an asyncio task that samples GPU metrics every sample_interval_ms.
        Safe to call multiple times - will not start duplicate tasks.
        """
        if self._sampling_task is not None and not self._sampling_task.done():
            logger.warning("event=sampling_already_running")
            return

        if not self.enabled:
            logger.warning("event=sampling_disabled reason='NVML not available'")
            return

        self._should_stop = False
        self._sampling_task = asyncio.create_task(self._sampling_loop())
        logger.info("event=sampling_task_created")

    async def stop_sampling(self) -> None:
        """
        Stop background sampling task.

        Gracefully stops the sampling loop and waits for task completion.
        """
        if self._sampling_task is None or self._sampling_task.done():
            logger.info("event=sampling_already_stopped")
            return

        logger.info("event=stopping_sampling")
        self._should_stop = True

        try:
            # Cancel task and wait for completion
            self._sampling_task.cancel()
            await asyncio.wait_for(self._sampling_task, timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("event=sampling_stop_timeout")
        except asyncio.CancelledError:
            pass

        self._sampling_task = None
        logger.info("event=sampling_stopped")

    def get_recent_samples(self, count: int = 100) -> List[TelemetrySample]:
        """
        Get recent telemetry samples from ring buffer.

        Args:
            count: Maximum number of recent samples to return

        Returns:
            List of recent TelemetrySample objects
        """
        return list(self.samples)[-count:]

    def get_last_sample(self) -> Optional[TelemetrySample]:
        """
        Get the most recent telemetry sample.

        Returns:
            Last TelemetrySample or None if no samples taken
        """
        return self._last_sample

    def get_device_name(self) -> str:
        """
        Get GPU device name.

        Returns:
            Device name string (e.g., "NVIDIA GeForce RTX 3090")
        """
        return self.device_name

    def get_stats(self) -> dict:
        """
        Get sampler statistics.

        Returns:
            Dictionary with sampler stats
        """
        return {
            "enabled": self.enabled,
            "device_name": self.device_name,
            "samples_collected": len(self.samples),
            "max_samples": self.max_samples,
            "sample_interval_ms": self.sample_interval_ms,
            "is_sampling": self._sampling_task is not None and not self._sampling_task.done(),
        }
