"""Configurable RFS-inspired set-evidence algorithm."""

from .algorithm import EvidenceResult, SetEvidenceAccumulator
from .config import RFSConfig, load_rfs_config
from .factory import DEFAULT_CONFIG_PATH, create_algorithm

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "EvidenceResult",
    "RFSConfig",
    "SetEvidenceAccumulator",
    "create_algorithm",
    "load_rfs_config",
]
