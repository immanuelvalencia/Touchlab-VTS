"""Paths and JSON persistence for Bayesian optimization artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Optional


BAYESIAN_DIR = Path(__file__).resolve().parent
DATASET_FILE = BAYESIAN_DIR / "touch_sequences.json"
ACTIVE_CONFIG_FILE = BAYESIAN_DIR / "config.json"
RUNS_DIR = BAYESIAN_DIR / "runs"
GENERATED_DATASETS_DIR = BAYESIAN_DIR / "generated_datasets"


@dataclass(frozen=True)
class OptimizationRunPaths:
    timestamp: str
    directory: Path
    dataset_snapshot: Path
    optimized_config: Path


def system_timestamp(now: Optional[datetime] = None) -> str:
    """Returns readable local system time using an explicit 24-hour clock."""
    local_time = now.astimezone() if now is not None else datetime.now().astimezone()
    return local_time.strftime("%Y-%m-%d_%H-%M-%S")


def build_run_paths(timestamp: Optional[str] = None) -> OptimizationRunPaths:
    """Creates system-local, dated paths for one optimization run."""
    value = timestamp or system_timestamp()
    directory = RUNS_DIR / value
    return OptimizationRunPaths(
        timestamp=value,
        directory=directory,
        dataset_snapshot=directory / f"touch_sequences_{value}.json",
        optimized_config=directory / f"bayesian_config_{value}.json",
    )


def build_generated_dataset_path(timestamp: Optional[str] = None) -> Path:
    value = timestamp or system_timestamp()
    return GENERATED_DATASETS_DIR / f"touch_sequences_random_{value}.json"


def write_json(path: Path, value: Any) -> None:
    """Writes JSON atomically so interruption cannot leave a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=4)
        handle.write("\n")
    os.replace(temporary_path, path)
