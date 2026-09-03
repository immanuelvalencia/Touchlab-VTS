"""Configuration loading for modified Naive Bayes tactile classification."""

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


def _nonnegative_row(value: Any, labels: Tuple[str, ...], name: str) -> dict[str, float]:
    row = _mapping(value, name)
    if set(row) != set(labels):
        raise ValueError(f"{name} must contain exactly the configured labels")
    values = {label: float(row[label]) for label in labels}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError(f"{name} values must be finite and nonnegative")
    return values


@dataclass(frozen=True)
class ModifiedNaiveBayesConfig:
    name: str
    features: Tuple[str, ...]
    shapes: Tuple[str, ...]
    aliases: Mapping[str, str]
    likelihoods: Mapping[str, Mapping[str, float]]
    presence: Mapping[str, Mapping[str, float]]
    feature_weights: Mapping[str, float]
    class_prior: Mapping[str, float]
    repeat_decay: float
    coverage_weight: float
    unexpected_penalty: float
    score_temperature: float
    coverage_rate: float
    source_path: Optional[Path] = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source_path: Optional[ConfigPath] = None,
    ) -> "ModifiedNaiveBayesConfig":
        root = _mapping(raw, "Modified Naive Bayes config")
        required = {
            "schema_version",
            "name",
            "features",
            "shapes",
            "likelihoods",
            "presence",
            "feature_weights",
            "class_prior",
            "parameters",
        }
        missing = sorted(required - set(root))
        if missing:
            raise ValueError("Modified Naive Bayes config is missing: " + ", ".join(missing))
        if root["schema_version"] != 1:
            raise ValueError("Modified Naive Bayes config requires schema_version 1")

        config_name = str(root["name"]).strip()
        if not config_name:
            raise ValueError("Modified Naive Bayes config name cannot be empty")
        features = _labels(root["features"], "features")
        shapes = _labels(root["shapes"], "shapes")
        aliases = {
            str(alias).strip().lower(): str(target).strip().lower()
            for alias, target in _mapping(root.get("aliases", {}), "aliases").items()
        }
        if any(not alias or target not in features for alias, target in aliases.items()):
            raise ValueError("Every Modified Naive Bayes alias must target a configured feature")

        likelihoods_raw = _mapping(root["likelihoods"], "likelihoods")
        if set(likelihoods_raw) != set(shapes):
            raise ValueError("likelihoods must contain exactly the configured shapes")
        likelihoods = {
            shape: _positive_row(likelihoods_raw[shape], features, f"likelihoods.{shape}")
            for shape in shapes
        }

        presence_raw = _mapping(root["presence"], "presence")
        if set(presence_raw) != set(shapes):
            raise ValueError("presence must contain exactly the configured shapes")
        presence = {}
        for shape in shapes:
            row = _nonnegative_row(presence_raw[shape], features, f"presence.{shape}")
            if any(value > 1.0 for value in row.values()):
                raise ValueError("presence values must lie in [0, 1]")
            presence[shape] = row

        feature_weights = _nonnegative_row(root["feature_weights"], features, "feature_weights")
        if any(value <= 0.0 for value in feature_weights.values()):
            raise ValueError("feature_weights must be positive")
        class_prior = _positive_row(root["class_prior"], shapes, "class_prior")

        parameters = _mapping(root["parameters"], "parameters")
        expected = {
            "repeat_decay",
            "coverage_weight",
            "unexpected_penalty",
            "score_temperature",
            "coverage_rate",
        }
        if set(parameters) != expected:
            raise ValueError(
                "Modified Naive Bayes parameters must contain only "
                + ", ".join(sorted(expected))
            )
        parsed = {name: float(parameters[name]) for name in expected}
        for name, value in parsed.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "score_temperature":
                if value <= 0.0:
                    raise ValueError("score_temperature must be positive")
            elif value < 0.0:
                raise ValueError(f"{name} must be nonnegative")

        return cls(
            name=config_name,
            features=features,
            shapes=shapes,
            aliases=aliases,
            likelihoods=likelihoods,
            presence=presence,
            feature_weights=feature_weights,
            class_prior=class_prior,
            repeat_decay=parsed["repeat_decay"],
            coverage_weight=parsed["coverage_weight"],
            unexpected_penalty=parsed["unexpected_penalty"],
            score_temperature=parsed["score_temperature"],
            coverage_rate=parsed["coverage_rate"],
            source_path=Path(source_path).resolve() if source_path else None,
        )


def load_config(path: ConfigPath) -> ModifiedNaiveBayesConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Modified Naive Bayes config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ModifiedNaiveBayesConfig.from_mapping(raw, source_path=config_path)
