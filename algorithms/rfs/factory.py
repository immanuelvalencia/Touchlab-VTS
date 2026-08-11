"""Runtime construction for configured set-evidence accumulators."""

from pathlib import Path
from typing import Optional, Union

from .algorithm import SetEvidenceAccumulator
from .config import load_rfs_config


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def create_algorithm(
    config_path: Optional[Union[str, Path]] = None,
) -> SetEvidenceAccumulator:
    selected_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return SetEvidenceAccumulator(load_rfs_config(selected_path))
