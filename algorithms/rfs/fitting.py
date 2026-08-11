"""Fit hard-label set-evidence templates from independent object trials."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, Mapping

from .config import RFSConfig


def _feature_label(value, config: RFSConfig) -> str:
    if not isinstance(value, str):
        raise ValueError("Every RFS touch must be one local-feature label string")
    label = value.strip().lower()
    label = config.aliases.get(label, label)
    if label not in config.features:
        raise ValueError(f"Unknown touch feature: {value!r}")
    return label


def fit_rfs_config(
    trials: Iterable[Mapping],
    base_config: RFSConfig,
    *,
    name: str,
    dirichlet_alpha: float = 1.0,
    presence_beta_alpha: float = 0.5,
    presence_beta_beta: float = 0.5,
) -> dict:
    """Learns feature-frequency and trial-presence templates from labels."""
    if dirichlet_alpha <= 0.0:
        raise ValueError("dirichlet_alpha must be greater than zero")
    if presence_beta_alpha <= 0.0 or presence_beta_beta <= 0.0:
        raise ValueError("Beta presence-prior parameters must be greater than zero")

    parsed_trials = defaultdict(list)
    for index, trial in enumerate(trials, start=1):
        if not isinstance(trial, Mapping):
            raise ValueError(f"Trial {index} must be an object")
        shape = str(trial.get("shape", "")).strip().lower()
        if shape not in base_config.shapes:
            raise ValueError(f"Trial {index} has unknown shape {shape!r}")
        sequence = trial.get("sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Trial {index} sequence must be a non-empty list")
        parsed_trials[shape].append(
            [_feature_label(touch, base_config) for touch in sequence]
        )

    missing_shapes = sorted(set(base_config.shapes) - set(parsed_trials))
    if missing_shapes:
        raise ValueError("No trials supplied for shapes: " + ", ".join(missing_shapes))

    feature_count = len(base_config.features)
    theta_templates: Dict[str, Dict[str, float]] = {}
    presence_templates: Dict[str, Dict[str, float]] = {}

    for shape in base_config.shapes:
        counts = {feature: dirichlet_alpha for feature in base_config.features}
        presence_counts = {feature: 0 for feature in base_config.features}
        for sequence in parsed_trials[shape]:
            for feature in sequence:
                counts[feature] += 1.0
            for feature in set(sequence):
                presence_counts[feature] += 1

        total_mass = sum(counts.values())
        trial_count = len(parsed_trials[shape])
        theta_templates[shape] = {
            feature: counts[feature] / total_mass
            for feature in base_config.features
        }
        presence_templates[shape] = {
            feature: (
                presence_counts[feature] + presence_beta_alpha
            ) / (
                trial_count + presence_beta_alpha + presence_beta_beta
            )
            for feature in base_config.features
        }

    return {
        "schema_version": 2,
        "name": name,
        "features": list(base_config.features),
        "shapes": list(base_config.shapes),
        "aliases": dict(base_config.aliases),
        "theta_templates": theta_templates,
        "presence_templates": presence_templates,
        "class_prior": {
            shape: 1.0 / len(base_config.shapes) for shape in base_config.shapes
        },
        "parameters": deepcopy(dict(base_config.parameters)),
        "fitting": {
            "input": "hard_local_feature_labels",
            "trial_count": sum(len(value) for value in parsed_trials.values()),
            "trial_count_by_shape": {
                shape: len(parsed_trials[shape]) for shape in base_config.shapes
            },
            "theta_prior": {
                "distribution": "symmetric_dirichlet",
                "alpha": dirichlet_alpha,
            },
            "presence_model": "trial_level_beta_bernoulli",
            "presence_prior": {
                "alpha": presence_beta_alpha,
                "beta": presence_beta_beta,
            },
        },
    }
