"""Dirichlet-Multinomial shape classifier."""

from .algorithm import DirichletMultinomialClassifier
from .factory import create_algorithm

__all__ = ["DirichletMultinomialClassifier", "create_algorithm"]
