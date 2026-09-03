"""Configuration loading for additive rule-based shape classification."""

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


def _weights(value: Any, features: Tuple[str, ...], name: str) -> dict[str, float]:
    row = _mapping(value, name)
    if set(row) != set(features):
        raise ValueError(f"{name} must contain exactly the configured features")
    weights = {feature: float(row[feature]) for feature in features}
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError(f"{name} weights must be finite and nonnegative")
    return weights


@dataclass(frozen=True)
class RuleBasedConfig:
    name: str
    features: Tuple[str, ...]
    shapes: Tuple[str, ...]
    aliases: Mapping[str, str]
    rules: Mapping[str, Mapping[str, Mapping[str, float]]]
    class_prior: Mapping[str, float]
    positive_weight: float
    negative_weight: float
    repetition_weight: float
    score_temperature: float
    source_path: Optional[Path] = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source_path: Optional[ConfigPath] = None,
    ) -> "RuleBasedConfig":
        root = _mapping(raw, "Rule-based config")
        required = {
            "schema_version",
            "name",
            "features",
            "shapes",
            "rules",
            "class_prior",
            "parameters",
        }
        missing = sorted(required - set(root))
        if missing:
            raise ValueError("Rule-based config is missing: " + ", ".join(missing))
        if root["schema_version"] != 1:
            raise ValueError("Rule-based config requires schema_version 1")

        name = str(root["name"]).strip()
        if not name:
            raise ValueError("Rule-based config name cannot be empty")
        features = _labels(root["features"], "features")
        shapes = _labels(root["shapes"], "shapes")
        aliases = {
            str(alias).strip().lower(): str(target).strip().lower()
            for alias, target in _mapping(root.get("aliases", {}), "aliases").items()
        }
        if any(not alias or target not in features for alias, target in aliases.items()):
            raise ValueError("Every rule-based alias must target a configured feature")

        rules_raw = _mapping(root["rules"], "rules")
        if set(rules_raw) != set(shapes):
            raise ValueError("rules must contain exactly the configured shapes")
        rules = {}
        for shape in shapes:
            rule = _mapping(rules_raw[shape], f"rules.{shape}")
            if set(rule) != {"positive", "negative"}:
                raise ValueError(f"rules.{shape} must contain positive and negative")
            rules[shape] = {
                "positive": _weights(rule["positive"], features, f"rules.{shape}.positive"),
                "negative": _weights(rule["negative"], features, f"rules.{shape}.negative"),
            }

        prior_raw = _mapping(root["class_prior"], "class_prior")
        if set(prior_raw) != set(shapes):
            raise ValueError("class_prior must contain exactly the configured shapes")
        prior_values = {shape: float(prior_raw[shape]) for shape in shapes}
        if any(not math.isfinite(value) or value <= 0.0 for value in prior_values.values()):
            raise ValueError("class_prior values must be finite and positive")
        prior_total = sum(prior_values.values())
        prior = {shape: value / prior_total for shape, value in prior_values.items()}

        parameters = _mapping(root["parameters"], "parameters")
        expected = {
            "positive_weight",
            "negative_weight",
            "repetition_weight",
            "score_temperature",
        }
        if set(parameters) != expected:
            raise ValueError(
                "Rule-based parameters must contain only "
                + ", ".join(sorted(expected))
            )
        positive_weight = float(parameters["positive_weight"])
        negative_weight = float(parameters["negative_weight"])
        repetition_weight = float(parameters["repetition_weight"])
        score_temperature = float(parameters["score_temperature"])
        for name, value in (
            ("positive_weight", positive_weight),
            ("negative_weight", negative_weight),
            ("repetition_weight", repetition_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(score_temperature) or score_temperature <= 0.0:
            raise ValueError("score_temperature must be finite and positive")

        return cls(
            name=name,
            features=features,
            shapes=shapes,
            aliases=aliases,
            rules=rules,
            class_prior=prior,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            repetition_weight=repetition_weight,
            score_temperature=score_temperature,
            source_path=Path(source_path).resolve() if source_path else None,
        )


def load_config(path: ConfigPath) -> RuleBasedConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Rule-based config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return RuleBasedConfig.from_mapping(raw, source_path=config_path)

