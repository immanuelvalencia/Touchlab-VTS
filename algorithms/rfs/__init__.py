"""Configurable RFS-inspired set-evidence algorithm."""

from .algorithm import EvidenceResult, SetEvidenceAccumulator
from .config import RFSConfig, load_rfs_config
from .factory import DEFAULT_CONFIG_PATH, create_algorithm
from .optimization import evaluate_frozen_config, optimize_rfs_config

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "EvidenceResult",
    "RFSConfig",
    "SetEvidenceAccumulator",
    "create_algorithm",
    "evaluate_frozen_config",
    "load_rfs_config",
    "optimize_rfs_config",
]
