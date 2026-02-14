"""Benchmark API routes."""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..bench.runner import BenchmarkRunner
from ..schemas.bench import BenchmarkRequest, BenchmarkResult, BenchmarkRunResponse
from ..storage.results_store import ResultsStore
from ..utils.env import get_environment_metadata

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory stores (use Redis/DB in production)
results_store = ResultsStore()
active_runs: Dict[str, BenchmarkResult] = {}


async def run_benchmark_task(run_id: str, request: BenchmarkRequest) -> None:
    """Background task to run benchmark."""
    try:
        logger.info(
            f"event=benchmark_start run_id={run_id} model={request.model_name} "
            f"engine={request.engine_type} batches={request.batch_sizes}"
        )
        active_runs[run_id].status = "running"

        runner = BenchmarkRunner()
        batch_results = await runner.run_benchmark(
            model_name=request.model_name,
            engine_type=request.engine_type,
            batch_sizes=request.batch_sizes,
            num_iterations=request.num_iterations,
            warmup_iterations=request.warmup_iterations,
        )

        # Update result
        active_runs[run_id].results = batch_results
        active_runs[run_id].status = "completed"
        active_runs[run_id].completed_at = datetime.now()

        # Store result
        results_store.save(run_id, active_runs[run_id])

        logger.info(
            f"event=benchmark_complete run_id={run_id} "
            f"results_count={len(batch_results)} duration_sec="
            f"{(active_runs[run_id].completed_at - active_runs[run_id].created_at).total_seconds():.2f}"
        )

    except Exception as e:
        logger.error(
            f"event=benchmark_failed run_id={run_id} error={str(e)}",
            exc_info=True
        )
        active_runs[run_id].status = "failed"
        active_runs[run_id].error_message = str(e)
        active_runs[run_id].completed_at = datetime.now()


@router.post("/run", response_model=BenchmarkRunResponse)
async def start_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks
) -> BenchmarkRunResponse:
    """
    Start a new benchmark run.

    The benchmark will run in the background. Use the returned run_id
    to poll for results via GET /bench/runs/{run_id}.
    """
    run_id = str(uuid.uuid4())

    logger.info(
        f"event=benchmark_request run_id={run_id} model={request.model_name} "
        f"engine={request.engine_type}"
    )

    # Create initial result object
    result = BenchmarkResult(
        run_id=run_id,
        model_name=request.model_name,
        engine_type=request.engine_type,
        environment=get_environment_metadata(),
        results=[],
        status="pending",
        created_at=datetime.now(),
    )

    active_runs[run_id] = result

    # Schedule background task
    background_tasks.add_task(run_benchmark_task, run_id, request)

    return BenchmarkRunResponse(
        run_id=run_id,
        status="pending",
        message=f"Benchmark run {run_id} started"
    )


@router.get("/runs/{run_id}", response_model=BenchmarkResult)
async def get_benchmark_result(run_id: str) -> BenchmarkResult:
    """
    Get benchmark result by run ID.

    Returns current status and results (if completed).
    Poll this endpoint to track benchmark progress.
    """
    logger.debug(f"event=get_result run_id={run_id}")

    # Check active runs first
    if run_id in active_runs:
        return active_runs[run_id]

    # Check storage
    result = results_store.get(run_id)
    if result is None:
        logger.warning(f"event=run_not_found run_id={run_id}")
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return result


@router.get("/runs", response_model=list[BenchmarkResult])
async def list_benchmark_runs(limit: int = 50) -> list[BenchmarkResult]:
    """List recent benchmark runs."""
    all_results = list(active_runs.values()) + results_store.list(limit=limit)
    # Sort by created_at descending
    all_results.sort(key=lambda x: x.created_at, reverse=True)
    return all_results[:limit]
