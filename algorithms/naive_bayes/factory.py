"""Runtime construction for Naive Bayes classifiers."""

from pathlib import Path
from typing import Optional, Union

from .algorithm import NaiveBayesClassifier
from .config import load_config


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def create_algorithm(
    config_path: Optional[Union[str, Path]] = None,
) -> NaiveBayesClassifier:
    selected_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return NaiveBayesClassifier(load_config(selected_path))

