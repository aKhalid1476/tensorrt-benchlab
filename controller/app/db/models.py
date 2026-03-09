"""SQLModel database models for controller."""
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class RunDB(SQLModel, table=True):
    """Database model for benchmark runs."""

    __tablename__ = "runs"

    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)

    # Run identification
    run_id: str = Field(unique=True, index=True)
    client_run_key: Optional[str] = Field(default=None, index=True)

    # Configuration
    runner_url: str
    model_name: str
    engines_json: str  # JSON-encoded list of engines
    batch_sizes_json: str  # JSON-encoded list of batch sizes
    num_iterations: int
    warmup_iterations: int

    # Status and timing
    status: str = Field(index=True)  # queued, running, succeeded, failed, cancelled
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results (stored as JSON)
    results_json: Optional[str] = None  # JSON-encoded list of EngineResult
    environment_json: Optional[str] = None  # JSON-encoded EnvironmentMetadata
    telemetry_json: Optional[str] = None  # JSON-encoded TelemetryResponse

    # Errors
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
