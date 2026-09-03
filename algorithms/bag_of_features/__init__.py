"""Bag-of-Features shape classifier."""

from .algorithm import BagOfFeaturesClassifier
from .factory import create_algorithm

__all__ = ["BagOfFeaturesClassifier", "create_algorithm"]
