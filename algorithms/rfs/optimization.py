"""Staged optimization for hard-label set-evidence shape classification."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import itertools
import math
import random
from statistics import mean
from typing import Iterable, Mapping, Sequence

from .algorithm import SetEvidenceAccumulator
from .config import RFSConfig
from .evaluation import evaluate_rfs_trials
from .fitting import fit_rfs_config


DEFAULT_DIRICHLET_ALPHAS = (0.05, 0.1, 0.25, 0.5, 1.0)
DEFAULT_PRESENCE_PRIORS = ((0.5, 0.5), (1.0, 1.0))
# With the fitted uniform class prior, absolute evidence scale is confounded with
# score temperature. Keep evidence at one and tune coverage relative to it by
# default; callers may explicitly supply an evidence ablation grid.
DEFAULT_EVIDENCE_WEIGHTS = (1.0,)
DEFAULT_COVERAGE_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
DEFAULT_UNEXPECTED_PENALTIES = (0.0, 0.1, 0.25, 0.5, 1.0)
DEFAULT_STOP_THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
DEFAULT_ENTROPY_THRESHOLDS = (0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20)
DEFAULT_MIN_TOUCHES = (1, 2, 3, 4, 5)
DEFAULT_STABILITY_WINDOWS = (1, 2, 3, 4)
DEFAULT_MAX_TOUCHES = (5, 6, 8, 10)


def _finite_positive(values: Sequence[float], name: str) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed or any(not math.isfinite(value) or value <= 0.0 for value in parsed):
        raise ValueError(f"{name} must contain finite values greater than zero")
    return parsed


def _finite_nonnegative(values: Sequence[float], name: str) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed or any(not math.isfinite(value) or value < 0.0 for value in parsed):
        raise ValueError(f"{name} must contain finite non-negative values")
    return parsed


def _positive_integers(values: Sequence[int], name: str) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed or any(value < 1 for value in parsed):
        raise ValueError(f"{name} must contain positive integers")
    return parsed


def _normalize_trials(
    trials: Iterable[Mapping],
    config: RFSConfig,
    *,
    name: str,
) -> list[dict]:
    normalized = []
    for index, trial in enumerate(trials, start=1):
        if not isinstance(trial, Mapping):
            raise ValueError(f"{name} trial {index} must be an object")
        shape = str(trial.get("shape", "")).strip().lower()
        if shape not in config.shapes:
            raise ValueError(f"{name} trial {index} has unknown shape {shape!r}")
        sequence = trial.get("sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"{name} trial {index} sequence must be a non-empty list")
        labels = []
        for touch in sequence:
            if not isinstance(touch, str):
                raise ValueError(
                    f"{name} trial {index} contains a non-string local feature"
                )
            label = touch.strip().lower()
            label = config.aliases.get(label, label)
            if label not in config.features:
                raise ValueError(
                    f"{name} trial {index} has unknown local feature {touch!r}"
                )
            labels.append(label)
        normalized.append({"shape": shape, "sequence": labels})
    if not normalized:
        raise ValueError(f"{name} trials cannot be empty")
    return normalized


def _ordered_sequences(
    trials: Sequence[Mapping],
    *,
    permutations: int,
    seed: int,
    max_touches: int,
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
                    "sequence": sequence[:max_touches],
                }
            )
    return runs


def _macro_prefix_metrics(
    config: RFSConfig,
    ordered_runs: Sequence[Mapping],
) -> dict:
    shape_run_metrics = defaultdict(list)
    for run in ordered_runs:
        accumulator = SetEvidenceAccumulator(config)
        losses = []
        briers = []
        correct = []
        for feature in run["sequence"]:
            result = accumulator.add_touch(feature)
            true_probability = max(result.belief[run["shape"]], config.parameters["eps"])
            losses.append(-math.log(true_probability))
            briers.append(
                sum(
                    (
                        probability
                        - (1.0 if shape == run["shape"] else 0.0)
                    )
                    ** 2
                    for shape, probability in result.belief.items()
                )
            )
            correct.append(result.prediction == run["shape"])
        shape_run_metrics[run["shape"]].append(
            {
                "nll": mean(losses),
                "brier": mean(briers),
                "accuracy": mean(correct),
            }
        )

    per_shape = {}
    for shape in config.shapes:
        values = shape_run_metrics.get(shape, [])
        if not values:
            raise ValueError(f"Validation trials do not include shape {shape!r}")
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


def _config_with_parameters(config_mapping: Mapping, **updates) -> RFSConfig:
    values = deepcopy(dict(config_mapping))
    values["parameters"] = deepcopy(dict(values["parameters"]))
    values["parameters"].update(updates)
    return RFSConfig.from_mapping(values)


def _calibrate_temperature(
    config_mapping: Mapping,
    ordered_runs: Sequence[Mapping],
    *,
    lower: float = 0.25,
    upper: float = 4.0,
    iterations: int = 32,
) -> tuple[float, dict]:
    """Minimizes macro prefix NLL over log temperature by golden-section search."""
    if lower <= 0.0 or upper <= lower:
        raise ValueError("Temperature bounds must satisfy 0 < lower < upper")
    left = math.log(lower)
    right = math.log(upper)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    cache = {}

    def evaluate(log_temperature: float) -> dict:
        key = round(log_temperature, 14)
        if key not in cache:
            temperature = math.exp(log_temperature)
            candidate = _config_with_parameters(
                config_mapping,
                score_temperature=temperature,
            )
            cache[key] = _macro_prefix_metrics(candidate, ordered_runs)
        return cache[key]

    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = evaluate(x1)
    f2 = evaluate(x2)
    for _ in range(iterations):
        if f1["macro_nll"] <= f2["macro_nll"]:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = evaluate(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = evaluate(x2)
    best_log, best_metrics = min(
        ((key, value) for key, value in cache.items()),
        key=lambda item: (item[1]["macro_nll"], item[1]["macro_brier"]),
    )
    return math.exp(best_log), best_metrics


def _build_trajectories(
    config: RFSConfig,
    ordered_runs: Sequence[Mapping],
) -> list[dict]:
    trajectories = []
    for run in ordered_runs:
        accumulator = SetEvidenceAccumulator(config)
        states = []
        for feature in run["sequence"]:
            result = accumulator.add_touch(feature)
            states.append(
                {
                    "prediction": result.prediction,
                    "confidence": result.belief[result.prediction],
                    "entropy": result.entropy,
                }
            )
        trajectories.append({"shape": run["shape"], "states": states})
    return trajectories


def _stopping_metrics(
    trajectories: Sequence[Mapping],
    *,
    stop_threshold: float,
    entropy_threshold: float,
    min_touches: int,
    stability_window: int,
    max_touches: int,
) -> dict:
    runs = []
    for trajectory in trajectories:
        states = trajectory["states"][:max_touches]
        accepted = False
        chosen = states[-1]
        chosen_touches = len(states)
        for index, state in enumerate(states):
            touch_count = index + 1
            start = index - stability_window + 1
            stable = start >= 0 and all(
                previous["prediction"] == state["prediction"]
                for previous in states[start : index + 1]
            )
            if (
                touch_count >= min_touches
                and state["confidence"] >= stop_threshold
                and state["entropy"] <= entropy_threshold
                and stable
            ):
                accepted = True
                chosen = state
                chosen_touches = touch_count
                break
        runs.append(
            {
                "accepted": accepted,
                "correct": chosen["prediction"] == trajectory["shape"],
                "touches": chosen_touches,
            }
        )
    accepted_runs = [run for run in runs if run["accepted"]]
    return {
        "overall_accuracy": mean(run["correct"] for run in runs),
        "acceptance_rate": mean(run["accepted"] for run in runs),
        "accepted_accuracy": (
            mean(run["correct"] for run in accepted_runs) if accepted_runs else None
        ),
        "uncertain_rate": mean(not run["accepted"] for run in runs),
        "mean_touches": mean(run["touches"] for run in runs),
    }


def optimize_rfs_config(
    train_trials: Iterable[Mapping],
    validation_trials: Iterable[Mapping],
    base_config: RFSConfig,
    *,
    name: str,
    permutations: int = 5,
    seed: int = 42,
    dirichlet_alphas: Sequence[float] = DEFAULT_DIRICHLET_ALPHAS,
    presence_priors: Sequence[tuple[float, float]] = DEFAULT_PRESENCE_PRIORS,
    evidence_weights: Sequence[float] = DEFAULT_EVIDENCE_WEIGHTS,
    coverage_weights: Sequence[float] = DEFAULT_COVERAGE_WEIGHTS,
    unexpected_penalties: Sequence[float] = DEFAULT_UNEXPECTED_PENALTIES,
    stop_thresholds: Sequence[float] = DEFAULT_STOP_THRESHOLDS,
    entropy_thresholds: Sequence[float] = DEFAULT_ENTROPY_THRESHOLDS,
    min_touches_values: Sequence[int] = DEFAULT_MIN_TOUCHES,
    stability_windows: Sequence[int] = DEFAULT_STABILITY_WINDOWS,
    max_touches_values: Sequence[int] = DEFAULT_MAX_TOUCHES,
    target_accepted_accuracy: float = 0.95,
    minimum_acceptance_rate: float = 0.80,
) -> tuple[dict, dict]:
    """Fits templates, tunes class scoring, calibrates belief, and tunes stopping."""
    if not 0.0 <= target_accepted_accuracy <= 1.0:
        raise ValueError("target_accepted_accuracy must lie in [0, 1]")
    if not 0.0 <= minimum_acceptance_rate <= 1.0:
        raise ValueError("minimum_acceptance_rate must lie in [0, 1]")
    dirichlet_alphas = _finite_positive(dirichlet_alphas, "dirichlet_alphas")
    evidence_weights = _finite_nonnegative(evidence_weights, "evidence_weights")
    coverage_weights = _finite_nonnegative(coverage_weights, "coverage_weights")
    unexpected_penalties = _finite_nonnegative(
        unexpected_penalties, "unexpected_penalties"
    )
    stop_thresholds = _finite_nonnegative(stop_thresholds, "stop_thresholds")
    entropy_thresholds = _finite_nonnegative(
        entropy_thresholds, "entropy_thresholds"
    )
    min_touches_values = _positive_integers(min_touches_values, "min_touches_values")
    stability_windows = _positive_integers(stability_windows, "stability_windows")
    max_touches_values = _positive_integers(max_touches_values, "max_touches_values")
    if any(value > 1.0 for value in stop_thresholds):
        raise ValueError("stop_thresholds must lie in [0, 1]")
    maximum_entropy = math.log(len(base_config.shapes))
    entropy_thresholds = tuple(
        value for value in entropy_thresholds if value <= maximum_entropy
    )
    if not entropy_thresholds:
        raise ValueError("No entropy threshold is valid for the configured shape count")

    parsed_priors = []
    for prior in presence_priors:
        if len(prior) != 2 or any(float(value) <= 0.0 for value in prior):
            raise ValueError("Every presence prior must contain two positive values")
        parsed_priors.append((float(prior[0]), float(prior[1])))
    if not parsed_priors:
        raise ValueError("presence_priors cannot be empty")

    train = _normalize_trials(train_trials, base_config, name="Training")
    validation = _normalize_trials(validation_trials, base_config, name="Validation")
    max_prefix_touches = max(max_touches_values)
    ordered_runs = _ordered_sequences(
        validation,
        permutations=permutations,
        seed=seed,
        max_touches=max_prefix_touches,
    )

    best_score = None
    score_candidates = 0
    for alpha, presence_prior in itertools.product(dirichlet_alphas, parsed_priors):
        fitted = fit_rfs_config(
            train,
            base_config,
            name=name,
            dirichlet_alpha=alpha,
            presence_beta_alpha=presence_prior[0],
            presence_beta_beta=presence_prior[1],
        )
        for lambda_evidence, lambda_coverage, unexpected_penalty in itertools.product(
            evidence_weights,
            coverage_weights,
            unexpected_penalties,
        ):
            candidate = _config_with_parameters(
                fitted,
                lambda_evidence=lambda_evidence,
                lambda_coverage=lambda_coverage,
                unexpected_penalty=unexpected_penalty,
                score_temperature=1.0,
                max_touches=max_prefix_touches,
            )
            metrics = _macro_prefix_metrics(candidate, ordered_runs)
            score_candidates += 1
            key = (
                metrics["macro_nll"],
                metrics["macro_brier"],
                -metrics["macro_accuracy"],
                lambda_evidence + lambda_coverage,
            )
            if best_score is None or key < best_score["key"]:
                best_score = {
                    "key": key,
                    "config": fitted,
                    "metrics": metrics,
                    "parameters": {
                        "dirichlet_alpha": alpha,
                        "presence_beta_alpha": presence_prior[0],
                        "presence_beta_beta": presence_prior[1],
                        "lambda_evidence": lambda_evidence,
                        "lambda_coverage": lambda_coverage,
                        "unexpected_penalty": unexpected_penalty,
                    },
                }

    selected_mapping = deepcopy(best_score["config"])
    selected_mapping["parameters"].update(
        {
            key: best_score["parameters"][key]
            for key in (
                "lambda_evidence",
                "lambda_coverage",
                "unexpected_penalty",
            )
        }
    )
    selected_mapping["parameters"]["max_touches"] = max_prefix_touches

    temperature, calibrated_metrics = _calibrate_temperature(
        selected_mapping,
        ordered_runs,
    )
    selected_mapping["parameters"]["score_temperature"] = temperature
    calibrated_config = RFSConfig.from_mapping(selected_mapping)
    trajectories = _build_trajectories(calibrated_config, ordered_runs)

    best_stopping = None
    stopping_candidates = 0
    for stop_threshold, entropy_threshold, min_touches, stability_window, max_touches in itertools.product(
        stop_thresholds,
        entropy_thresholds,
        min_touches_values,
        stability_windows,
        max_touches_values,
    ):
        if min_touches > max_touches or stability_window > max_touches:
            continue
        metrics = _stopping_metrics(
            trajectories,
            stop_threshold=stop_threshold,
            entropy_threshold=entropy_threshold,
            min_touches=min_touches,
            stability_window=stability_window,
            max_touches=max_touches,
        )
        stopping_candidates += 1
        accepted_accuracy = metrics["accepted_accuracy"]
        accuracy_shortfall = (
            target_accepted_accuracy
            if accepted_accuracy is None
            else max(0.0, target_accepted_accuracy - accepted_accuracy)
        )
        acceptance_shortfall = max(
            0.0, minimum_acceptance_rate - metrics["acceptance_rate"]
        )
        feasible = accuracy_shortfall == 0.0 and acceptance_shortfall == 0.0
        if feasible:
            key = (
                0,
                metrics["mean_touches"],
                -metrics["acceptance_rate"],
                -metrics["overall_accuracy"],
                stop_threshold,
            )
        else:
            key = (
                1,
                accuracy_shortfall + acceptance_shortfall,
                -metrics["overall_accuracy"],
                metrics["mean_touches"],
                -metrics["acceptance_rate"],
            )
        if best_stopping is None or key < best_stopping["key"]:
            best_stopping = {
                "key": key,
                "feasible": feasible,
                "metrics": metrics,
                "parameters": {
                    "stop_threshold": stop_threshold,
                    "entropy_threshold": entropy_threshold,
                    "min_touches": min_touches,
                    "stability_window": stability_window,
                    "max_touches": max_touches,
                },
            }

    selected_mapping["parameters"].update(best_stopping["parameters"])
    selected_mapping["name"] = name
    selected_mapping["optimization"] = {
        "input": "deterministic_hard_local_feature_labels",
        "seed": seed,
        "validation_permutations": permutations,
        "score_objective": "macro_mean_prefix_negative_log_likelihood",
        "score_candidates": score_candidates,
        "stopping_candidates": stopping_candidates,
        "target_accepted_accuracy": target_accepted_accuracy,
        "minimum_acceptance_rate": minimum_acceptance_rate,
        "stopping_constraints_satisfied": best_stopping["feasible"],
    }
    report = {
        "name": name,
        "training_trials": len(train),
        "validation_trials": len(validation),
        "validation_runs": len(ordered_runs),
        "selected_template_and_score_parameters": best_score["parameters"],
        "validation_before_temperature_calibration": best_score["metrics"],
        "selected_score_temperature": temperature,
        "temperature_search_bounds": [0.25, 4.0],
        "temperature_at_search_boundary": (
            temperature <= 0.2501 or temperature >= 3.999
        ),
        "validation_after_temperature_calibration": calibrated_metrics,
        "selected_stopping_parameters": best_stopping["parameters"],
        "validation_stopping_metrics": best_stopping["metrics"],
        "stopping_constraints_satisfied": best_stopping["feasible"],
        "search": selected_mapping["optimization"],
    }
    return selected_mapping, report


def evaluate_frozen_config(
    trials: Iterable[Mapping],
    config: RFSConfig,
    *,
    permutations: int = 10,
    seed: int = 42,
) -> dict:
    """Evaluates a frozen configuration without performing parameter selection."""
    parsed = _normalize_trials(trials, config, name="Test")
    return evaluate_rfs_trials(parsed, config, permutations=permutations, seed=seed)
