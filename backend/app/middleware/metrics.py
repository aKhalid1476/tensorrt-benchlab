"""Middleware for tracking API request metrics."""
import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..telemetry.prometheus_metrics import record_api_request

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track API request metrics.

    Automatically records:
    - Request latency (histogram)
    - Request count (counter)
    - HTTP method, endpoint, and status code
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request and record metrics.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        # Skip metrics endpoint to avoid infinite loop
        if request.url.path == "/metrics":
            return await call_next(request)

        # Record start time
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Record error
            status_code = 500
            logger.error(
                f"event=request_error path={request.url.path} error={str(e)}",
                exc_info=True
            )
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Record metrics
            record_api_request(
                method=request.method,
                endpoint=request.url.path,
                status=status_code,
                duration=duration,
            )

        return response
