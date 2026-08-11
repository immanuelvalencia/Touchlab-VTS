"""Evaluation and validation-only stopping-rule tuning for set evidence."""

from __future__ import annotations

import random
from statistics import mean
from typing import Iterable, Mapping

from .algorithm import SetEvidenceAccumulator
from .config import RFSConfig


def _validated_trials(trials: Iterable[Mapping], config: RFSConfig) -> list[dict]:
    parsed = []
    for index, trial in enumerate(trials, start=1):
        if not isinstance(trial, Mapping):
            raise ValueError(f"Trial {index} must be an object")
        shape = str(trial.get("shape", "")).strip().lower()
        sequence = trial.get("sequence")
        if shape not in config.shapes:
            raise ValueError(f"Trial {index} has unknown shape {shape!r}")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Trial {index} sequence must be a non-empty list")
        parsed.append({"shape": shape, "sequence": sequence})
    if not parsed:
        raise ValueError("At least one trial is required")
    return parsed


def evaluate_rfs_trials(
    trials: Iterable[Mapping],
    config: RFSConfig,
    *,
    permutations: int = 10,
    seed: int = 42,
) -> dict:
    """Evaluates final predictions, stopping behavior, and touch-count curves."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    parsed = _validated_trials(trials, config)
    rng = random.Random(seed)
    runs = []
    curve = {touch_count: [] for touch_count in range(1, config.parameters["max_touches"] + 1)}

    for trial_index, trial in enumerate(parsed):
        for repeat in range(permutations):
            sequence = list(trial["sequence"])
            rng.shuffle(sequence)
            sequence = sequence[: config.parameters["max_touches"]]
            accumulator = SetEvidenceAccumulator(config)
            first_stop = None
            last_result = None
            for touch_count, touch in enumerate(sequence, start=1):
                result = accumulator.add_touch(touch)
                last_result = result
                curve[touch_count].append(result.prediction == trial["shape"])
                if first_stop is None and result.should_stop:
                    first_stop = result

            stopping_result = first_stop or last_result
            accepted = bool(
                first_stop is not None and first_stop.stopping_reason == "confidence"
            )
            runs.append(
                {
                    "trial_index": trial_index,
                    "permutation": repeat,
                    "true_shape": trial["shape"],
                    "prediction": stopping_result.prediction,
                    "correct": stopping_result.prediction == trial["shape"],
                    "accepted": accepted,
                    "uncertain": not accepted,
                    "touches": stopping_result.touch_count,
                    "confidence": stopping_result.belief[stopping_result.prediction],
                    "entropy": stopping_result.entropy,
                    "stopping_reason": stopping_result.stopping_reason or "sequence_exhausted",
                }
            )

    accepted_runs = [run for run in runs if run["accepted"]]
    touch_curve = {
        str(touch_count): {
            "runs": len(values),
            "accuracy": mean(values),
        }
        for touch_count, values in curve.items()
        if values
    }
    return {
        "trial_count": len(parsed),
        "permutations_per_trial": permutations,
        "run_count": len(runs),
        "overall_accuracy": mean(run["correct"] for run in runs),
        "acceptance_rate": mean(run["accepted"] for run in runs),
        "accepted_accuracy": (
            mean(run["correct"] for run in accepted_runs) if accepted_runs else None
        ),
        "uncertain_rate": mean(run["uncertain"] for run in runs),
        "mean_touches": mean(run["touches"] for run in runs),
        "accuracy_by_touch_count": touch_curve,
        "runs": runs,
    }
