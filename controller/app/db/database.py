"""Database connection and session management."""
import logging
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from .models import RunDB

logger = logging.getLogger(__name__)

# Database URL (SQLite)
DATABASE_DIR = Path("./data")
DATABASE_DIR.mkdir(exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/benchlab.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL logging
    connect_args={"check_same_thread": False},  # Needed for SQLite
)


def init_db() -> None:
    """Initialize database (create tables)."""
    SQLModel.metadata.create_all(engine)
    logger.info(f"event=db_initialized url={DATABASE_URL}")


def get_session() -> Session:
    """Get database session."""
    return Session(engine)
