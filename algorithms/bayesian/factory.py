"""Runtime construction for Bayesian shape predictors."""

from pathlib import Path

from .algorithm import BayesianShapePredictor


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def create_algorithm(config_path=None) -> BayesianShapePredictor:
    return BayesianShapePredictor(config_path or DEFAULT_CONFIG_PATH)
