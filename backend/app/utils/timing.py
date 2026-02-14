"""Timing utilities for benchmarks."""
import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer() -> Iterator[dict]:
    """Context manager for timing code blocks."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
