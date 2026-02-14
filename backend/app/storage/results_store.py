"""Results storage (in-memory for now)."""
import json
from pathlib import Path
from typing import List, Optional

from ..schemas.bench import BenchmarkResult


class ResultsStore:
    """In-memory results store with JSON persistence."""

    def __init__(self, storage_dir: str = "./data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.cache: dict[str, BenchmarkResult] = {}

    def save(self, run_id: str, result: BenchmarkResult) -> None:
        """Save benchmark result."""
        self.cache[run_id] = result

        # Persist to disk
        file_path = self.storage_dir / f"{run_id}.json"
        with open(file_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

    def get(self, run_id: str) -> Optional[BenchmarkResult]:
        """Get benchmark result by ID."""
        # Check cache first
        if run_id in self.cache:
            return self.cache[run_id]

        # Check disk
        file_path = self.storage_dir / f"{run_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
                result = BenchmarkResult(**data)
                self.cache[run_id] = result
                return result

        return None

    def list(self, limit: int = 50) -> List[BenchmarkResult]:
        """List recent benchmark results."""
        results = []
        for file_path in sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]:
            with open(file_path, "r") as f:
                data = json.load(f)
                results.append(BenchmarkResult(**data))

        return results
