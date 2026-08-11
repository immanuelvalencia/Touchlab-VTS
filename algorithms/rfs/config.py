"""Configuration loading for hard-label set evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union


ConfigPath = Union[str, Path]

PARAMETER_NAMES = {
    "lambda_evidence",
    "lambda_coverage",
    "unexpected_penalty",
    "score_temperature",
    "stop_threshold",
    "entropy_threshold",
    "min_touches",
    "stability_window",
    "max_touches",
    "eps",
}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_labels(value: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    labels = tuple(str(item).strip().lower() for item in value)
    if not labels or any(not label for label in labels):
        raise ValueError(f"{name} must contain non-empty labels")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} must contain unique labels")
    return labels


@dataclass(frozen=True)
class RFSConfig:
    """Shape templates and parameters for hard-label set evidence."""

    name: str
    schema_version: int
    features: Tuple[str, ...]
    shapes: Tuple[str, ...]
    aliases: Mapping[str, str]
    theta_templates: Mapping[str, Mapping[str, float]]
    presence_templates: Mapping[str, Mapping[str, float]]
    class_prior: Optional[Mapping[str, float]]
    parameters: Mapping[str, Any]
    source_path: Optional[Path] = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source_path: Optional[ConfigPath] = None,
    ) -> "RFSConfig":
        root = _require_mapping(raw, "RFS config")
        required = {
            "schema_version",
            "name",
            "features",
            "shapes",
            "theta_templates",
            "presence_templates",
            "parameters",
        }
        missing = sorted(required - set(root))
        if missing:
            raise ValueError(f"RFS config is missing required keys: {', '.join(missing)}")

        schema_version = root["schema_version"]
        if schema_version != 2:
            raise ValueError(
                f"Unsupported RFS config schema_version: {schema_version!r}; "
                "hard-label set evidence requires version 2"
            )

        name = str(root["name"]).strip()
        if not name:
            raise ValueError("RFS config name cannot be empty")
        features = _require_labels(root["features"], "features")
        shapes = _require_labels(root["shapes"], "shapes")
        if len(features) < 2:
            raise ValueError("features must contain at least two labels")

        aliases_raw = _require_mapping(root.get("aliases", {}), "aliases")
        aliases: Dict[str, str] = {}
        for alias, target in aliases_raw.items():
            normalized_alias = str(alias).strip().lower()
            normalized_target = str(target).strip().lower()
            if not normalized_alias or normalized_target not in features:
                raise ValueError(f"Invalid feature alias {alias!r}: {target!r}")
            aliases[normalized_alias] = normalized_target

        theta_raw = _require_mapping(root["theta_templates"], "theta_templates")
        theta = {
            str(shape): dict(_require_mapping(row, f"theta_templates.{shape}"))
            for shape, row in theta_raw.items()
        }
        presence_raw = _require_mapping(root["presence_templates"], "presence_templates")
        presence = {
            str(shape): dict(_require_mapping(row, f"presence_templates.{shape}"))
            for shape, row in presence_raw.items()
        }

        parameters = dict(_require_mapping(root["parameters"], "parameters"))
        missing_parameters = sorted(PARAMETER_NAMES - set(parameters))
        unknown_parameters = sorted(set(parameters) - PARAMETER_NAMES)
        if missing_parameters:
            raise ValueError(
                "RFS config is missing parameters: " + ", ".join(missing_parameters)
            )
        if unknown_parameters:
            raise ValueError(
                "RFS config contains unknown parameters: " + ", ".join(unknown_parameters)
            )

        class_prior = root.get("class_prior")
        if class_prior is not None:
            class_prior = dict(_require_mapping(class_prior, "class_prior"))

        return cls(
            name=name,
            schema_version=schema_version,
            features=features,
            shapes=shapes,
            aliases=aliases,
            theta_templates=theta,
            presence_templates=presence,
            class_prior=class_prior,
            parameters=parameters,
            source_path=Path(source_path).resolve() if source_path else None,
        )


def load_rfs_config(path: ConfigPath) -> RFSConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"RFS config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in RFS config {config_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    return RFSConfig.from_mapping(raw, source_path=config_path)
