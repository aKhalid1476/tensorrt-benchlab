"""NVML GPU telemetry sampler."""
import logging
from collections import deque
from datetime import datetime
from typing import List

from ..schemas.bench import TelemetrySample

logger = logging.getLogger(__name__)


class NVMLSampler:
    """GPU telemetry sampler using NVML."""

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.samples: deque[TelemetrySample] = deque(maxlen=max_samples)
        self.device_name = "Unknown"

        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.device_name = pynvml.nvmlDeviceGetName(self.handle)
            self.enabled = True
            logger.info(f"NVML initialized for device: {self.device_name}")
        except Exception as e:
            logger.warning(f"NVML initialization failed: {e}")
            self.enabled = False
            self.handle = None

    def sample(self) -> TelemetrySample:
        """Take a single telemetry sample."""
        if not self.enabled:
            return TelemetrySample(
                timestamp=datetime.now(),
                gpu_utilization_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
            )

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
            return sample

        except Exception as e:
            logger.error(f"Failed to sample GPU: {e}")
            return TelemetrySample(
                timestamp=datetime.now(),
                gpu_utilization_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
            )

    def get_recent_samples(self, count: int = 100) -> List[TelemetrySample]:
        """Get recent telemetry samples."""
        # Sample current state
        self.sample()

        # Return recent samples
        return list(self.samples)[-count:]

    def get_device_name(self) -> str:
        """Get GPU device name."""
        return self.device_name
