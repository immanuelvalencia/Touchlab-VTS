"""Configuration loading for the hard-label Bag-of-Features classifier."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union


ConfigPath = Union[str, Path]


def _labels(value: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    labels = tuple(str(item).strip().lower() for item in value)
    if not labels or any(not label for label in labels):
        raise ValueError(f"{name} must contain non-empty labels")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} must contain unique labels")
    return labels


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


@dataclass(frozen=True)
class BagOfFeaturesConfig:
    name: str
    features: Tuple[str, ...]
    shapes: Tuple[str, ...]
    aliases: Mapping[str, str]
    histogram_templates: Mapping[str, Mapping[str, float]]
    class_prior: Mapping[str, float]
    distance_temperature: float
    source_path: Optional[Path] = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source_path: Optional[ConfigPath] = None,
    ) -> "BagOfFeaturesConfig":
        root = _mapping(raw, "Bag-of-Features config")
        required = {
            "schema_version",
            "name",
            "features",
            "shapes",
            "histogram_templates",
            "class_prior",
            "parameters",
        }
        missing = sorted(required - set(root))
        if missing:
            raise ValueError("Bag-of-Features config is missing: " + ", ".join(missing))
        if root["schema_version"] != 1:
            raise ValueError("Bag-of-Features config requires schema_version 1")

        features = _labels(root["features"], "features")
        shapes = _labels(root["shapes"], "shapes")
        name = str(root["name"]).strip()
        if not name:
            raise ValueError("Bag-of-Features config name cannot be empty")

        aliases = {
            str(alias).strip().lower(): str(target).strip().lower()
            for alias, target in _mapping(root.get("aliases", {}), "aliases").items()
        }
        if any(not alias or target not in features for alias, target in aliases.items()):
            raise ValueError("Every Bag-of-Features alias must target a configured feature")

        templates_raw = _mapping(root["histogram_templates"], "histogram_templates")
        if set(templates_raw) != set(shapes):
            raise ValueError("histogram_templates must contain exactly the configured shapes")
        templates = {}
        for shape in shapes:
            row = _mapping(templates_raw[shape], f"histogram_templates.{shape}")
            if set(row) != set(features):
                raise ValueError(f"Histogram template for {shape!r} has a feature mismatch")
            values = {feature: float(row[feature]) for feature in features}
            if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
                raise ValueError("Histogram template values must be finite and nonnegative")
            total = sum(values.values())
            if total <= 0.0:
                raise ValueError(f"Histogram template for {shape!r} has zero mass")
            templates[shape] = {feature: value / total for feature, value in values.items()}

        prior_raw = _mapping(root["class_prior"], "class_prior")
        if set(prior_raw) != set(shapes):
            raise ValueError("class_prior must contain exactly the configured shapes")
        prior_values = {shape: float(prior_raw[shape]) for shape in shapes}
        if any(not math.isfinite(value) or value <= 0.0 for value in prior_values.values()):
            raise ValueError("class_prior values must be finite and positive")
        prior_total = sum(prior_values.values())
        prior = {shape: value / prior_total for shape, value in prior_values.items()}

        parameters = _mapping(root["parameters"], "parameters")
        if set(parameters) != {"distance_temperature"}:
            raise ValueError("Bag-of-Features parameters must contain only distance_temperature")
        temperature = float(parameters["distance_temperature"])
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("distance_temperature must be finite and positive")

        return cls(
            name=name,
            features=features,
            shapes=shapes,
            aliases=aliases,
            histogram_templates=templates,
            class_prior=prior,
            distance_temperature=temperature,
            source_path=Path(source_path).resolve() if source_path else None,
        )


def load_config(path: ConfigPath) -> BagOfFeaturesConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Bag-of-Features config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return BagOfFeaturesConfig.from_mapping(raw, source_path=config_path)
