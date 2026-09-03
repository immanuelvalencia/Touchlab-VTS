"""Runtime construction for Bag-of-Features classifiers."""

from pathlib import Path
from typing import Optional, Union

from .algorithm import BagOfFeaturesClassifier
from .config import load_config


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def create_algorithm(
    config_path: Optional[Union[str, Path]] = None,
) -> BagOfFeaturesClassifier:
    selected_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return BagOfFeaturesClassifier(load_config(selected_path))
