"""Telemetry and metrics API routes."""
import logging

from fastapi import APIRouter

from ..schemas.bench import TelemetryResponse
from ..telemetry.nvml_sampler import NVMLSampler
from ..telemetry.prometheus_metrics import update_gpu_metrics

router = APIRouter()
logger = logging.getLogger(__name__)

# Global NVML sampler instance
sampler = NVMLSampler()


def get_sampler() -> NVMLSampler:
    """
    Get global NVML sampler instance.

    Returns:
        NVMLSampler instance
    """
    return sampler


def update_prometheus_gpu_metrics() -> None:
    """
    Update Prometheus GPU metrics from latest sample.

    This function is called periodically to export GPU metrics to Prometheus.
    """
    last_sample = sampler.get_last_sample()
    if last_sample is None:
        return

    update_gpu_metrics(
        device_name=sampler.get_device_name(),
        utilization=last_sample.gpu_utilization_percent,
        memory_used=last_sample.memory_used_mb,
        memory_total=last_sample.memory_total_mb,
        temperature=last_sample.temperature_celsius,
        power=last_sample.power_usage_watts,
    )


@router.get("/live", response_model=TelemetryResponse)
async def get_live_telemetry() -> TelemetryResponse:
    """
    Get recent GPU telemetry samples.

    Returns up to 100 most recent telemetry samples from the ring buffer.
    Also updates Prometheus metrics with the latest sample.
    """
    # Update Prometheus metrics
    update_prometheus_gpu_metrics()

    samples = sampler.get_recent_samples(count=100)
    device_name = sampler.get_device_name()

    return TelemetryResponse(
        samples=samples,
        device_name=device_name
    )


@router.get("/stats")
async def get_sampler_stats() -> dict:
    """
    Get sampler statistics.

    Returns information about the telemetry sampler status.
    """
    return sampler.get_stats()
