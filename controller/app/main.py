"""FastAPI controller service (Mac M2 compatible)."""
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, List, Optional

import httpx
from contracts import (
    EngineType,
    RunCreateRequest,
    RunCreateResponse,
    RunListResponse,
    RunRecord,
    RunStatus,
    RunnerExecuteRequest,
    TelemetryResponse,
)
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import REGISTRY, generate_latest
from sqlmodel import Session, select

from .db.database import get_session, init_db
from .db.models import RunDB
from .metrics import (
    runs_total,
    runs_by_status,
    run_duration_seconds,
    runner_request_duration_seconds,
    runner_request_retries_total,
    active_runs,
)
from .reports import generate_markdown_report

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan."""
    logger.info("service=controller event=startup")
    init_db()
    yield
    logger.info("service=controller event=shutdown")


app = FastAPI(
    title="TensorRT BenchLab Controller",
    version="0.1.0",
    description="Benchmark orchestration service (Mac M2 compatible)",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(f"path={request.url.path} error={str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check."""
    return {"status": "healthy", "service": "tensorrt-benchlab-controller"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


async def call_runner_with_retry(
    url: str,
    payload: dict,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    run_id: str = "",
    engine_type: str = "unknown",
) -> dict:
    """
    Call runner API with exponential backoff retry logic.

    Args:
        url: Runner endpoint URL
        payload: Request payload
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff duration in seconds
        run_id: Run ID for logging
        engine_type: Engine type for metrics labeling

    Returns:
        Response JSON

    Raises:
        httpx.HTTPStatusError: If all retries fail
    """
    backoff = initial_backoff
    start_time = time.time()

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()

                # Record successful request duration
                duration = time.time() - start_time
                runner_request_duration_seconds.labels(
                    engine=engine_type, status="success"
                ).observe(duration)

                return result

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            is_last_attempt = (attempt == max_retries - 1)

            # Record retry attempt
            if attempt > 0:
                runner_request_retries_total.labels(
                    engine=engine_type, attempt=str(attempt + 1)
                ).inc()

            logger.warning(
                f"event=runner_call_failed run_id={run_id} attempt={attempt + 1}/{max_retries} "
                f"error={str(e)} will_retry={not is_last_attempt}"
            )

            if is_last_attempt:
                # Record failed request duration
                duration = time.time() - start_time
                runner_request_duration_seconds.labels(
                    engine=engine_type, status="failure"
                ).observe(duration)
                raise

            # Exponential backoff
            logger.info(
                f"event=retry_backoff run_id={run_id} backoff_sec={backoff} "
                f"next_attempt={attempt + 2}/{max_retries}"
            )
            await asyncio.sleep(backoff)
            backoff *= 2  # Exponential backoff: 1s, 2s, 4s...


async def execute_run_on_runner(run_id: str, db_run: RunDB) -> None:
    """Background task to execute run on runner."""
    logger.info(f"event=execute_start run_id={run_id}")

    session = get_session()
    run_start_time = time.time()

    try:
        # Merge the db_run into this session
        db_run = session.merge(db_run)

        # Update status to running
        db_run.status = RunStatus.RUNNING.value
        db_run.started_at = datetime.now()
        session.add(db_run)
        session.commit()

        # Track active runs
        active_runs.inc()
        runs_by_status.labels(status="running").inc()
        runs_by_status.labels(status="queued").dec()

        # Parse configuration
        engines = json.loads(db_run.engines_json)
        batch_sizes = json.loads(db_run.batch_sizes_json)

        # Execute on runner for each engine
        all_results = []

        for engine_str in engines:
            # Check for cancellation before each engine
            # Query fresh from database instead of refreshing
            statement = select(RunDB).where(RunDB.run_id == run_id)
            current_run = session.exec(statement).first()
            if current_run and current_run.status == RunStatus.CANCELLED.value:
                logger.info(f"event=run_cancelled run_id={run_id} engine={engine_str}")
                return

            engine_type = EngineType(engine_str)

            # Call runner with retry logic
            runner_request = RunnerExecuteRequest(
                run_id=run_id,
                model_name=db_run.model_name,
                engine_type=engine_type,
                batch_sizes=batch_sizes,
                num_iterations=db_run.num_iterations,
                warmup_iterations=db_run.warmup_iterations,
            )

            logger.info(
                f"event=call_runner run_id={run_id} engine={engine_type} url={db_run.runner_url}"
            )

            # Call with retry logic (max 3 attempts, exponential backoff)
            runner_response = await call_runner_with_retry(
                url=f"{db_run.runner_url}/execute",
                payload=runner_request.model_dump(),
                max_retries=3,
                initial_backoff=1.0,
                run_id=run_id,
                engine_type=engine_type.value,
            )

            # Collect results
            if runner_response["status"] == "succeeded":
                all_results.extend(runner_response["results"])

                # Store environment and telemetry from first successful run
                if db_run.environment_json is None:
                    db_run.environment_json = json.dumps(runner_response["environment"])
                if db_run.telemetry_json is None:
                    db_run.telemetry_json = json.dumps(runner_response["telemetry"])
            else:
                # Runner failed
                raise RuntimeError(
                    f"Runner failed: {runner_response.get('error_message', 'Unknown error')}"
                )

        # All engines succeeded
        db_run.status = RunStatus.SUCCEEDED.value
        db_run.results_json = json.dumps(all_results)
        db_run.completed_at = datetime.now()

        # Save model_name before closing session
        model_name = db_run.model_name

        session.add(db_run)
        session.commit()
        session.close()

        # Record metrics
        run_duration = time.time() - run_start_time
        runs_total.labels(model=model_name, status="succeeded").inc()
        runs_by_status.labels(status="succeeded").inc()
        runs_by_status.labels(status="running").dec()
        run_duration_seconds.labels(model=model_name, status="succeeded").observe(
            run_duration
        )
        active_runs.dec()

        logger.info(
            f"event=execute_complete run_id={run_id} results_count={len(all_results)} "
            f"duration_sec={run_duration:.2f}"
        )

    except Exception as e:
        logger.error(f"event=execute_failed run_id={run_id} error={str(e)}", exc_info=True)

        # Update status to failed
        import traceback

        db_run.status = RunStatus.FAILED.value
        db_run.error_message = str(e)
        db_run.error_stack = traceback.format_exc()
        db_run.completed_at = datetime.now()

        # Save model_name before closing session
        model_name = db_run.model_name

        session.add(db_run)
        session.commit()
        session.close()

        # Record metrics
        run_duration = time.time() - run_start_time
        runs_total.labels(model=model_name, status="failed").inc()
        runs_by_status.labels(status="failed").inc()
        runs_by_status.labels(status="running").dec()
        run_duration_seconds.labels(model=model_name, status="failed").observe(
            run_duration
        )
        active_runs.dec()


@app.post("/runs", response_model=RunCreateResponse)
async def create_run(
    request: RunCreateRequest, background_tasks: BackgroundTasks
) -> RunCreateResponse:
    """
    Create a new benchmark run.

    Supports idempotency via client_run_key:
    - If client_run_key is provided and matches an existing run, returns existing run_id
    - Otherwise creates a new run
    """
    session = get_session()

    # Idempotency check
    if request.client_run_key:
        statement = select(RunDB).where(RunDB.client_run_key == request.client_run_key)
        existing_run = session.exec(statement).first()

        if existing_run:
            logger.info(
                f"event=idempotent_run_found run_id={existing_run.run_id} "
                f"client_run_key={request.client_run_key}"
            )
            session.close()
            return RunCreateResponse(
                run_id=existing_run.run_id,
                status=RunStatus(existing_run.status),
                message=f"Existing run {existing_run.run_id} returned (idempotent)",
            )

    # Create new run
    run_id = str(uuid.uuid4())

    logger.info(
        f"event=create_run run_id={run_id} model={request.model_name} "
        f"engines={request.engines} client_run_key={request.client_run_key or 'none'}"
    )

    # Create database record
    db_run = RunDB(
        run_id=run_id,
        client_run_key=request.client_run_key,
        runner_url=request.runner_url,
        model_name=request.model_name,
        engines_json=json.dumps([e.value for e in request.engines]),
        batch_sizes_json=json.dumps(request.batch_sizes),
        num_iterations=request.num_iterations,
        warmup_iterations=request.warmup_iterations,
        status=RunStatus.QUEUED.value,
    )

    session.add(db_run)
    session.commit()
    session.close()

    # Track queued run
    runs_by_status.labels(status="queued").inc()

    # Schedule background execution
    background_tasks.add_task(execute_run_on_runner, run_id, db_run)

    return RunCreateResponse(
        run_id=run_id, status=RunStatus.QUEUED, message=f"Run {run_id} created"
    )


@app.get("/runs/{run_id}", response_model=RunRecord)
async def get_run(run_id: str) -> RunRecord:
    """Get run by ID."""
    session = get_session()
    statement = select(RunDB).where(RunDB.run_id == run_id)
    db_run = session.exec(statement).first()
    session.close()

    if not db_run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Convert DB model to RunRecord
    engines = [EngineType(e) for e in json.loads(db_run.engines_json)]
    batch_sizes = json.loads(db_run.batch_sizes_json)
    results = json.loads(db_run.results_json) if db_run.results_json else []
    environment = json.loads(db_run.environment_json) if db_run.environment_json else None
    telemetry = json.loads(db_run.telemetry_json) if db_run.telemetry_json else None

    return RunRecord(
        run_id=db_run.run_id,
        status=RunStatus(db_run.status),
        created_at=db_run.created_at,
        started_at=db_run.started_at,
        completed_at=db_run.completed_at,
        runner_url=db_run.runner_url,
        model_name=db_run.model_name,
        engines=engines,
        batch_sizes=batch_sizes,
        num_iterations=db_run.num_iterations,
        warmup_iterations=db_run.warmup_iterations,
        client_run_key=db_run.client_run_key,
        environment=environment,
        results=results,
        telemetry=telemetry,
        error_message=db_run.error_message,
        error_stack=db_run.error_stack,
    )


@app.get("/runs", response_model=RunListResponse)
async def list_runs(limit: int = 50) -> RunListResponse:
    """List runs."""
    session = get_session()
    statement = select(RunDB).order_by(RunDB.created_at.desc()).limit(limit)
    db_runs = session.exec(statement).all()
    session.close()

    runs = []
    for db_run in db_runs:
        engines = [EngineType(e) for e in json.loads(db_run.engines_json)]
        batch_sizes = json.loads(db_run.batch_sizes_json)
        results = json.loads(db_run.results_json) if db_run.results_json else []
        environment = (
            json.loads(db_run.environment_json) if db_run.environment_json else None
        )
        telemetry = json.loads(db_run.telemetry_json) if db_run.telemetry_json else None

        runs.append(
            RunRecord(
                run_id=db_run.run_id,
                status=RunStatus(db_run.status),
                created_at=db_run.created_at,
                started_at=db_run.started_at,
                completed_at=db_run.completed_at,
                runner_url=db_run.runner_url,
                model_name=db_run.model_name,
                engines=engines,
                batch_sizes=batch_sizes,
                num_iterations=db_run.num_iterations,
                warmup_iterations=db_run.warmup_iterations,
                client_run_key=db_run.client_run_key,
                environment=environment,
                results=results,
                telemetry=telemetry,
                error_message=db_run.error_message,
                error_stack=db_run.error_stack,
            )
        )

    return RunListResponse(runs=runs, total=len(runs))


@app.get("/runs/{run_id}/report.md")
async def get_run_report_markdown(run_id: str, save: bool = False) -> dict[str, str]:
    """
    Generate markdown report for a run.

    Args:
        run_id: Run identifier
        save: If True, save report to disk

    Returns:
        Dictionary with markdown content and optional file path
    """
    # Get run data
    run_record = await get_run(run_id)

    # Generate markdown
    markdown = generate_markdown_report(run_record)

    response = {"markdown": markdown}

    # Optionally save to disk
    if save:
        import os

        reports_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        file_path = os.path.join(reports_dir, f"{run_id}.md")
        with open(file_path, "w") as f:
            f.write(markdown)

        response["file_path"] = file_path
        logger.info(f"event=report_saved run_id={run_id} path={file_path}")

    return response


@app.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, str]:
    """
    Cancel a running or queued benchmark run.

    Only works for runs in QUEUED or RUNNING status.
    Once cancelled, the run cannot be resumed.
    """
    session = get_session()
    statement = select(RunDB).where(RunDB.run_id == run_id)
    db_run = session.exec(statement).first()

    if not db_run:
        session.close()
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Check if run can be cancelled
    current_status = RunStatus(db_run.status)
    if current_status not in [RunStatus.QUEUED, RunStatus.RUNNING]:
        session.close()
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel run in {current_status} status. "
            f"Only queued or running runs can be cancelled.",
        )

    # Update status to cancelled
    db_run.status = RunStatus.CANCELLED.value
    db_run.completed_at = datetime.now()

    session.add(db_run)
    session.commit()
    session.close()

    logger.info(
        f"event=run_cancelled run_id={run_id} previous_status={current_status.value}"
    )

    return {
        "run_id": run_id,
        "status": "cancelled",
        "message": f"Run {run_id} cancelled successfully",
    }
