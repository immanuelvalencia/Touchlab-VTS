"""Folder-based shape algorithms and their central registry."""

from .registry import (
    AlgorithmSpec,
    get_algorithm,
    get_algorithm_by_display_name,
    get_algorithm_specs,
    register_algorithm,
)

__all__ = [
    "AlgorithmSpec",
    "get_algorithm",
    "get_algorithm_by_display_name",
    "get_algorithm_specs",
    "register_algorithm",
]
