"""Shared evaluation and artifact helpers for hard-label shape classifiers."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
from statistics import mean
from typing import Callable, Iterable, Mapping, Sequence


def load_trials(path: Path) -> list[dict]:
    selected = Path(path).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"Trial file not found: {selected}")
    with selected.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Trial file must contain a JSON array: {selected}")
    return value


def normalize_trials(
    trials: Iterable[Mapping],
    *,
    features: Sequence[str],
    shapes: Sequence[str],
    aliases: Mapping[str, str],
    name: str,
) -> list[dict]:
    normalized = []
    for index, trial in enumerate(trials, start=1):
        if not isinstance(trial, Mapping):
            raise ValueError(f"{name} trial {index} must be an object")
        shape = str(trial.get("shape", "")).strip().lower()
        if shape not in shapes:
            raise ValueError(f"{name} trial {index} has unknown shape {shape!r}")
        sequence = trial.get("sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"{name} trial {index} sequence must be a non-empty list")
        labels = []
        for value in sequence:
            if not isinstance(value, str):
                raise ValueError(f"{name} trial {index} contains a non-string feature")
            label = value.strip().lower()
            label = aliases.get(label, label)
            if label not in features:
                raise ValueError(
                    f"{name} trial {index} contains unknown feature {value!r}"
                )
            labels.append(label)
        normalized.append({"shape": shape, "sequence": labels})
    if not normalized:
        raise ValueError(f"{name} trials cannot be empty")
    missing = sorted(set(shapes) - {trial["shape"] for trial in normalized})
    if missing:
        raise ValueError(f"{name} trials do not include shapes: {', '.join(missing)}")
    return normalized


def ordered_runs(
    trials: Sequence[Mapping], *, permutations: int, seed: int
) -> list[dict]:
    if permutations < 1:
        raise ValueError("permutations must be positive")
    rng = random.Random(seed)
    runs = []
    for trial_index, trial in enumerate(trials):
        original = list(trial["sequence"])
        for repeat in range(permutations):
            sequence = list(original)
            if repeat > 0:
                rng.shuffle(sequence)
            runs.append(
                {
                    "trial_index": trial_index,
                    "shape": trial["shape"],
                    "sequence": sequence,
                }
            )
    return runs


def evaluate_prefixes(
    runs: Sequence[Mapping],
    *,
    shapes: Sequence[str],
    create_classifier: Callable[[], object],
) -> dict:
    by_shape = defaultdict(list)
    final_by_shape = defaultdict(list)
    by_touch_count = defaultdict(list)
    run_records = []
    for run in runs:
        classifier = create_classifier()
        prefix_nll = []
        prefix_brier = []
        prefix_correct = []
        probabilities = None
        for touch_count, feature in enumerate(run["sequence"], start=1):
            probabilities = classifier.update(feature)
            true_probability = max(float(probabilities[run["shape"]]), 1e-15)
            prefix_nll.append(-math.log(true_probability))
            prefix_brier.append(
                sum(
                    (
                        probability
                        - (1.0 if shape == run["shape"] else 0.0)
                    )
                    ** 2
                    for shape, probability in probabilities.items()
                )
            )
            prefix_correct.append(max(probabilities, key=probabilities.get) == run["shape"])
            by_touch_count[touch_count].append(prefix_correct[-1])
        by_shape[run["shape"]].append(
            {
                "nll": mean(prefix_nll),
                "brier": mean(prefix_brier),
                "accuracy": mean(prefix_correct),
            }
        )
        final_by_shape[run["shape"]].append(
            {
                "nll": -math.log(max(float(probabilities[run["shape"]]), 1e-15)),
                "brier": sum(
                    (
                        probability
                        - (1.0 if shape == run["shape"] else 0.0)
                    )
                    ** 2
                    for shape, probability in probabilities.items()
                ),
                "accuracy": max(probabilities, key=probabilities.get) == run["shape"],
            }
        )
        run_records.append(
            {
                "trial_index": run["trial_index"],
                "true_shape": run["shape"],
                "prediction": max(probabilities, key=probabilities.get),
                "correct": max(probabilities, key=probabilities.get) == run["shape"],
                "touches": len(run["sequence"]),
                "belief": dict(probabilities),
            }
        )

    def summarize(source) -> dict:
        per_shape = {}
        for shape in shapes:
            values = source[shape]
            per_shape[shape] = {
                key: mean(value[key] for value in values)
                for key in ("nll", "brier", "accuracy")
            }
        return {
            "macro_nll": mean(value["nll"] for value in per_shape.values()),
            "macro_brier": mean(value["brier"] for value in per_shape.values()),
            "macro_accuracy": mean(value["accuracy"] for value in per_shape.values()),
            "per_shape": per_shape,
        }

    confusion = {
        true_shape: {predicted_shape: 0 for predicted_shape in shapes}
        for true_shape in shapes
    }
    for record in run_records:
        confusion[record["true_shape"]][record["prediction"]] += 1
    return {
        "prefix": summarize(by_shape),
        "final": summarize(final_by_shape),
        "accuracy_by_touch_count": {
            str(touch_count): {
                "runs": len(values),
                "accuracy": mean(values),
            }
            for touch_count, values in sorted(by_touch_count.items())
        },
        "confusion_matrix": confusion,
        "runs": run_records,
    }


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")


def write_json(path: Path, value) -> Path:
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_suffix(selected.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, selected)
    return selected


def install_model_sidecar(config_path: Path, model_path: Path, suffix: str) -> Path:
    model = Path(model_path).expanduser().resolve()
    if model.suffix.lower() != ".pth":
        raise ValueError("--model must point to a .pth weights file")
    if not model.is_file():
        raise FileNotFoundError(f"Model weights not found: {model}")
    sidecar = model.with_suffix(suffix)
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    shutil.copyfile(config_path, temporary)
    os.replace(temporary, sidecar)
    return sidecar


def install_algorithm_default(config_path: Path, default_config_path: Path) -> tuple[Path, Path | None]:
    """Atomically install an optimized config as an algorithm's default config."""
    source = Path(config_path).expanduser().resolve()
    destination = Path(default_config_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Optimized config not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    backup = None
    if destination.is_file():
        backup = destination.with_name(
            f"config_backup_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copyfile(destination, backup)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    return destination, backup
