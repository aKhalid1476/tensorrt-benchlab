"""Telemetry and metrics API routes."""
import logging

from fastapi import APIRouter

from ..schemas.bench import TelemetryResponse
from ..telemetry.nvml_sampler import NVMLSampler

router = APIRouter()
logger = logging.getLogger(__name__)

sampler = NVMLSampler()


@router.get("/live", response_model=TelemetryResponse)
async def get_live_telemetry() -> TelemetryResponse:
    """Get recent GPU telemetry samples."""
    samples = sampler.get_recent_samples(count=100)
    device_name = sampler.get_device_name()

    return TelemetryResponse(
        samples=samples,
        device_name=device_name
    )
