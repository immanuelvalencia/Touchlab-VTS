"""Validation and atomic persistence for hard-label RFS touch trials."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TRIAL_SPLITS = ("train", "validation", "test")


def trial_file(root, split: str) -> Path:
    normalized_split = str(split).strip().lower()
    if normalized_split not in TRIAL_SPLITS:
        raise ValueError(f"split must be one of {TRIAL_SPLITS}")
    return Path(root).expanduser().resolve() / f"rfs_{normalized_split}.json"


def normalize_feature_label(
    value: str,
    features: Sequence[str],
    aliases: Mapping[str, str],
) -> str:
    if not isinstance(value, str):
        raise ValueError("Every RFS touch must be one local-feature label string")
    label = value.strip().lower()
    label = aliases.get(label, label)
    if label not in features:
        raise ValueError(f"Feature label is not configured for RFS: {value!r}")
    return label


def load_trials(path) -> list[dict]:
    selected_path = Path(path)
    if not selected_path.is_file():
        return []
    try:
        with selected_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid trial JSON in {selected_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, list):
        raise ValueError(f"Trial file must contain a JSON array: {selected_path}")
    return value


def save_trials(path, trials: Iterable[Mapping]) -> Path:
    """Atomically replaces one trial JSON file with shape/sequence records."""
    selected_path = Path(path).expanduser().resolve()
    serialized = []
    for index, trial in enumerate(trials, start=1):
        if not isinstance(trial, Mapping):
            raise ValueError(f"Trial {index} must be an object")
        shape = str(trial.get("shape", "")).strip().lower()
        sequence = trial.get("sequence")
        if not shape:
            raise ValueError(f"Trial {index} shape cannot be empty")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Trial {index} sequence must be a non-empty list")
        if any(not isinstance(feature, str) or not feature.strip() for feature in sequence):
            raise ValueError(f"Trial {index} contains an invalid feature label")
        serialized.append(
            {
                "shape": shape,
                "sequence": [feature.strip().lower() for feature in sequence],
            }
        )

    selected_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected_path.with_suffix(selected_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=4)
        handle.write("\n")
    os.replace(temporary, selected_path)
    return selected_path


def append_trial(
    root,
    *,
    split: str,
    shape: str,
    sequence: Iterable[str],
    features: Sequence[str],
    aliases: Mapping[str, str],
) -> tuple[Path, int]:
    normalized_shape = str(shape).strip().lower()
    if not normalized_shape:
        raise ValueError("shape cannot be empty")
    normalized_sequence = [
        normalize_feature_label(feature, features, aliases)
        for feature in sequence
    ]
    if not normalized_sequence:
        raise ValueError("A trial must contain at least one touch")

    path = trial_file(root, split)
    trials = load_trials(path)
    trials.append({"shape": normalized_shape, "sequence": normalized_sequence})
    save_trials(path, trials)
    return path, len(trials)


def trial_counts(root) -> dict[str, int]:
    return {
        split: len(load_trials(trial_file(root, split)))
        for split in TRIAL_SPLITS
    }
