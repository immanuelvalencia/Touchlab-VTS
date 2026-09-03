"""Configuration loading for Naive Bayes tactile shape classification."""

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


def _positive_row(value: Any, labels: Tuple[str, ...], name: str) -> dict[str, float]:
    row = _mapping(value, name)
    if set(row) != set(labels):
        raise ValueError(f"{name} must contain exactly the configured labels")
    values = {label: float(row[label]) for label in labels}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError(f"{name} values must be finite and positive")
    total = sum(values.values())
    return {label: value / total for label, value in values.items()}


@dataclass(frozen=True)
class NaiveBayesConfig:
    name: str
    features: Tuple[str, ...]
    shapes: Tuple[str, ...]
    aliases: Mapping[str, str]
    likelihoods: Mapping[str, Mapping[str, float]]
    class_prior: Mapping[str, float]
    score_temperature: float
    source_path: Optional[Path] = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source_path: Optional[ConfigPath] = None,
    ) -> "NaiveBayesConfig":
        root = _mapping(raw, "Naive Bayes config")
        required = {
            "schema_version",
            "name",
            "features",
            "shapes",
            "likelihoods",
            "class_prior",
            "parameters",
        }
        missing = sorted(required - set(root))
        if missing:
            raise ValueError("Naive Bayes config is missing: " + ", ".join(missing))
        if root["schema_version"] != 1:
            raise ValueError("Naive Bayes config requires schema_version 1")

        name = str(root["name"]).strip()
        if not name:
            raise ValueError("Naive Bayes config name cannot be empty")
        features = _labels(root["features"], "features")
        shapes = _labels(root["shapes"], "shapes")
        aliases = {
            str(alias).strip().lower(): str(target).strip().lower()
            for alias, target in _mapping(root.get("aliases", {}), "aliases").items()
        }
        if any(not alias or target not in features for alias, target in aliases.items()):
            raise ValueError("Every Naive Bayes alias must target a configured feature")

        likelihoods_raw = _mapping(root["likelihoods"], "likelihoods")
        if set(likelihoods_raw) != set(shapes):
            raise ValueError("likelihoods must contain exactly the configured shapes")
        likelihoods = {
            shape: _positive_row(likelihoods_raw[shape], features, f"likelihoods.{shape}")
            for shape in shapes
        }

        prior = _positive_row(root["class_prior"], shapes, "class_prior")

        parameters = _mapping(root["parameters"], "parameters")
        if set(parameters) != {"score_temperature"}:
            raise ValueError("Naive Bayes parameters must contain only score_temperature")
        score_temperature = float(parameters["score_temperature"])
        if not math.isfinite(score_temperature) or score_temperature <= 0.0:
            raise ValueError("score_temperature must be finite and positive")

        return cls(
            name=name,
            features=features,
            shapes=shapes,
            aliases=aliases,
            likelihoods=likelihoods,
            class_prior=prior,
            score_temperature=score_temperature,
            source_path=Path(source_path).resolve() if source_path else None,
        )


def load_config(path: ConfigPath) -> NaiveBayesConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Naive Bayes config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return NaiveBayesConfig.from_mapping(raw, source_path=config_path)

