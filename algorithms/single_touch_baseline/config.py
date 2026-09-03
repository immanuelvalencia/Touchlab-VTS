"""Configuration loading for the single-touch baseline classifier."""

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


def _probability_row(row: Mapping[str, Any], labels: Tuple[str, ...], name: str) -> dict[str, float]:
    if set(row) != set(labels):
        raise ValueError(f"{name} must contain exactly the configured shapes")
    values = {label: float(row[label]) for label in labels}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError(f"{name} values must be finite and positive")
    total = sum(values.values())
    return {label: value / total for label, value in values.items()}


@dataclass(frozen=True)
class SingleTouchBaselineConfig:
    name: str
    features: Tuple[str, ...]
    shapes: Tuple[str, ...]
    aliases: Mapping[str, str]
    feature_posteriors: Mapping[str, Mapping[str, float]]
    score_temperature: float
    source_path: Optional[Path] = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source_path: Optional[ConfigPath] = None,
    ) -> "SingleTouchBaselineConfig":
        root = _mapping(raw, "Single-touch baseline config")
        required = {
            "schema_version",
            "name",
            "features",
            "shapes",
            "feature_posteriors",
            "parameters",
        }
        missing = sorted(required - set(root))
        if missing:
            raise ValueError("Single-touch baseline config is missing: " + ", ".join(missing))
        if root["schema_version"] != 1:
            raise ValueError("Single-touch baseline config requires schema_version 1")

        name = str(root["name"]).strip()
        if not name:
            raise ValueError("Single-touch baseline config name cannot be empty")
        features = _labels(root["features"], "features")
        shapes = _labels(root["shapes"], "shapes")
        aliases = {
            str(alias).strip().lower(): str(target).strip().lower()
            for alias, target in _mapping(root.get("aliases", {}), "aliases").items()
        }
        if any(not alias or target not in features for alias, target in aliases.items()):
            raise ValueError("Every single-touch alias must target a configured feature")

        posterior_raw = _mapping(root["feature_posteriors"], "feature_posteriors")
        if set(posterior_raw) != set(features):
            raise ValueError("feature_posteriors must contain exactly the configured features")
        posteriors = {
            feature: _probability_row(
                _mapping(posterior_raw[feature], f"feature_posteriors.{feature}"),
                shapes,
                f"feature_posteriors.{feature}",
            )
            for feature in features
        }

        parameters = _mapping(root["parameters"], "parameters")
        if set(parameters) != {"score_temperature"}:
            raise ValueError("Single-touch parameters must contain only score_temperature")
        temperature = float(parameters["score_temperature"])
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("score_temperature must be finite and positive")

        return cls(
            name=name,
            features=features,
            shapes=shapes,
            aliases=aliases,
            feature_posteriors=posteriors,
            score_temperature=temperature,
            source_path=Path(source_path).resolve() if source_path else None,
        )


def load_config(path: ConfigPath) -> SingleTouchBaselineConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Single-touch baseline config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return SingleTouchBaselineConfig.from_mapping(raw, source_path=config_path)

