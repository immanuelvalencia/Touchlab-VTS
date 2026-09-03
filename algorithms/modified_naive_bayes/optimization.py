"""Fit and optimize Modified Naive Bayes from hard-label touch trials."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys

from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.hard_label_optimization import (
    evaluate_prefixes,
    install_model_sidecar,
    load_trials,
    normalize_trials,
    ordered_runs,
    timestamp,
    write_json,
)
from algorithms.modified_naive_bayes.algorithm import ModifiedNaiveBayesClassifier
from algorithms.modified_naive_bayes.config import ModifiedNaiveBayesConfig, load_config
from algorithms.modified_naive_bayes.factory import DEFAULT_CONFIG_PATH


DEFAULT_TRAIN = PROJECT_ROOT / "trials" / "rfs_train.json"
DEFAULT_VALIDATION = PROJECT_ROOT / "trials" / "rfs_validation.json"
DEFAULT_RUNS = Path(__file__).resolve().parent / "runs"


def _float_list(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from exc
    if not parsed or any(item < 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("Values must be nonnegative")
    return parsed


def _fit_likelihoods(trials, config, smoothing: float) -> dict:
    likelihoods = {}
    for shape in config.shapes:
        counts = {feature: smoothing for feature in config.features}
        for trial in trials:
            if trial["shape"] != shape:
                continue
            for feature in trial["sequence"]:
                counts[feature] += 1.0
        total = sum(counts.values())
        likelihoods[shape] = {feature: counts[feature] / total for feature in config.features}
    return likelihoods


def _fit_presence(trials, config) -> dict:
    presence = {
        shape: {feature: 0.0 for feature in config.features}
        for shape in config.shapes
    }
    for trial in trials:
        for feature in set(trial["sequence"]):
            presence[trial["shape"]][feature] = 1.0
    return presence


def _fit_prior(trials, config) -> dict:
    counts = {shape: 1.0 for shape in config.shapes}
    for trial in trials:
        counts[trial["shape"]] += 1.0
    total = sum(counts.values())
    return {shape: counts[shape] / total for shape in config.shapes}


def _feature_weights(config, likelihoods, class_prior, strength: float) -> dict:
    weights = {}
    maximum_entropy = math.log(len(config.shapes))
    for feature in config.features:
        masses = [
            class_prior[shape] * likelihoods[shape][feature]
            for shape in config.shapes
        ]
        total = sum(masses)
        if total <= 0.0:
            weights[feature] = 1.0
            continue
        posterior = [mass / total for mass in masses]
        entropy = -sum(value * math.log(value) for value in posterior if value > 0.0)
        discriminability = 1.0 - entropy / maximum_entropy
        weights[feature] = max(0.05, 1.0 + strength * discriminability)
    return weights


def _mapping(
    config,
    *,
    name: str,
    likelihoods,
    presence,
    feature_weights,
    class_prior,
    repeat_decay: float,
    coverage_weight: float,
    unexpected_penalty: float,
    score_temperature: float,
    coverage_rate: float,
) -> dict:
    return {
        "schema_version": 1,
        "name": name,
        "features": list(config.features),
        "shapes": list(config.shapes),
        "aliases": dict(config.aliases),
        "likelihoods": deepcopy(likelihoods),
        "presence": deepcopy(presence),
        "feature_weights": deepcopy(feature_weights),
        "class_prior": dict(class_prior),
        "parameters": {
            "repeat_decay": repeat_decay,
            "coverage_weight": coverage_weight,
            "unexpected_penalty": unexpected_penalty,
            "score_temperature": score_temperature,
            "coverage_rate": coverage_rate,
        },
    }


def optimize(
    train,
    validation,
    base,
    *,
    smoothing_values,
    repeat_decay_values,
    coverage_weight_values,
    unexpected_penalty_values,
    feature_weight_strength_values,
    coverage_rate_values,
    permutations,
    seed,
    name,
):
    parsed_train = normalize_trials(
        train,
        features=base.features,
        shapes=base.shapes,
        aliases=base.aliases,
        name="Training",
    )
    parsed_validation = normalize_trials(
        validation,
        features=base.features,
        shapes=base.shapes,
        aliases=base.aliases,
        name="Validation",
    )
    runs = ordered_runs(parsed_validation, permutations=permutations, seed=seed)
    class_prior = _fit_prior(parsed_train, base)
    presence = _fit_presence(parsed_train, base)
    candidates = []

    for smoothing in smoothing_values:
        likelihoods = _fit_likelihoods(parsed_train, base, smoothing)
        for feature_weight_strength in feature_weight_strength_values:
            feature_weights = _feature_weights(base, likelihoods, class_prior, feature_weight_strength)
            for repeat_decay in repeat_decay_values:
                for coverage_weight in coverage_weight_values:
                    for unexpected_penalty in unexpected_penalty_values:
                        for coverage_rate in coverage_rate_values:

                            def objective(log_temperature):
                                mapping = _mapping(
                                    base,
                                    name=name,
                                    likelihoods=likelihoods,
                                    presence=presence,
                                    feature_weights=feature_weights,
                                    class_prior=class_prior,
                                    repeat_decay=repeat_decay,
                                    coverage_weight=coverage_weight,
                                    unexpected_penalty=unexpected_penalty,
                                    score_temperature=math.exp(log_temperature),
                                    coverage_rate=coverage_rate,
                                )
                                config = ModifiedNaiveBayesConfig.from_mapping(mapping)
                                metrics = evaluate_prefixes(
                                    runs,
                                    shapes=config.shapes,
                                    create_classifier=lambda: ModifiedNaiveBayesClassifier(config),
                                )
                                return metrics["prefix"]["macro_nll"]

                            result = minimize_scalar(
                                objective,
                                bounds=(math.log(0.05), math.log(8.0)),
                                method="bounded",
                                options={"xatol": 1e-4},
                            )
                            temperature = math.exp(result.x)
                            mapping = _mapping(
                                base,
                                name=name,
                                likelihoods=likelihoods,
                                presence=presence,
                                feature_weights=feature_weights,
                                class_prior=class_prior,
                                repeat_decay=repeat_decay,
                                coverage_weight=coverage_weight,
                                unexpected_penalty=unexpected_penalty,
                                score_temperature=temperature,
                                coverage_rate=coverage_rate,
                            )
                            config = ModifiedNaiveBayesConfig.from_mapping(mapping)
                            metrics = evaluate_prefixes(
                                runs,
                                shapes=config.shapes,
                                create_classifier=lambda: ModifiedNaiveBayesClassifier(config),
                            )
                            candidates.append(
                                {
                                    "key": (
                                        metrics["prefix"]["macro_nll"],
                                        metrics["prefix"]["macro_brier"],
                                        -metrics["prefix"]["macro_accuracy"],
                                    ),
                                    "smoothing": smoothing,
                                    "feature_weight_strength": feature_weight_strength,
                                    "repeat_decay": repeat_decay,
                                    "coverage_weight": coverage_weight,
                                    "unexpected_penalty": unexpected_penalty,
                                    "coverage_rate": coverage_rate,
                                    "temperature": temperature,
                                    "mapping": mapping,
                                    "metrics": metrics,
                                    "temperature_optimizer_success": bool(result.success),
                                }
                            )

    best = min(candidates, key=lambda candidate: candidate["key"])
    report = {
        "method": "modified_naive_bayes_tactile_evidence",
        "selection_metric": "class-balanced mean validation prefix NLL",
        "training_trial_count": len(parsed_train),
        "validation_trial_count": len(parsed_validation),
        "validation_permutations": permutations,
        "seed": seed,
        "smoothing_candidates": list(smoothing_values),
        "repeat_decay_candidates": list(repeat_decay_values),
        "coverage_weight_candidates": list(coverage_weight_values),
        "unexpected_penalty_candidates": list(unexpected_penalty_values),
        "feature_weight_strength_candidates": list(feature_weight_strength_values),
        "coverage_rate_candidates": list(coverage_rate_values),
        "selected_smoothing": best["smoothing"],
        "selected_feature_weight_strength": best["feature_weight_strength"],
        "selected_repeat_decay": best["repeat_decay"],
        "selected_coverage_weight": best["coverage_weight"],
        "selected_unexpected_penalty": best["unexpected_penalty"],
        "selected_coverage_rate": best["coverage_rate"],
        "selected_score_temperature": best["temperature"],
        "validation_metrics": best["metrics"],
        "candidate_count": len(candidates),
    }
    return best["mapping"], report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--smoothing", type=_float_list, default=(0.01, 0.05, 0.1, 0.5, 1.0))
    parser.add_argument("--repeat-decay", type=_float_list, default=(0.0, 0.5, 1.0))
    parser.add_argument("--coverage-weight", type=_float_list, default=(0.0, 0.5, 1.0))
    parser.add_argument("--unexpected-penalty", type=_float_list, default=(0.0, 0.5, 1.0))
    parser.add_argument("--feature-weight-strength", type=_float_list, default=(0.0, 0.5, 1.0))
    parser.add_argument("--coverage-rate", type=_float_list, default=(0.5, 1.0, 2.0))
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--test-permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base = load_config(args.base_config)
    train = load_trials(args.train)
    validation = load_trials(args.validation)
    run_timestamp = timestamp()
    optimized, report = optimize(
        train,
        validation,
        base,
        smoothing_values=args.smoothing,
        repeat_decay_values=args.repeat_decay,
        coverage_weight_values=args.coverage_weight,
        unexpected_penalty_values=args.unexpected_penalty,
        feature_weight_strength_values=args.feature_weight_strength,
        coverage_rate_values=args.coverage_rate,
        permutations=args.permutations,
        seed=args.seed,
        name=f"{base.name}_optimized_{run_timestamp}",
    )
    run_dir = args.output_dir.expanduser().resolve() / run_timestamp
    config_path = write_json(run_dir / f"modified_naive_bayes_config_{run_timestamp}.json", optimized)
    write_json(run_dir / f"train_{run_timestamp}.json", train)
    write_json(run_dir / f"validation_{run_timestamp}.json", validation)
    if args.test:
        test = load_trials(args.test)
        parsed_test = normalize_trials(
            test,
            features=base.features,
            shapes=base.shapes,
            aliases=base.aliases,
            name="Test",
        )
        runs = ordered_runs(parsed_test, permutations=args.test_permutations, seed=args.seed)
        config = ModifiedNaiveBayesConfig.from_mapping(optimized, source_path=config_path)
        report["test_metrics"] = evaluate_prefixes(
            runs,
            shapes=config.shapes,
            create_classifier=lambda: ModifiedNaiveBayesClassifier(config),
        )
        write_json(run_dir / f"test_{run_timestamp}.json", test)

    report["artifacts"] = {"config": str(config_path), "run_directory": str(run_dir)}
    report_path = write_json(run_dir / f"optimization_report_{run_timestamp}.json", report)
    sidecar = install_model_sidecar(config_path, args.model, ".mnb.json") if args.model else None
    prefix = report["validation_metrics"]["prefix"]
    print(f"Selected smoothing:         {report['selected_smoothing']}")
    print(f"Selected repeat decay:      {report['selected_repeat_decay']}")
    print(f"Selected coverage weight:   {report['selected_coverage_weight']}")
    print(f"Selected unexpected penalty:{report['selected_unexpected_penalty']}")
    print(f"Score temperature:          {report['selected_score_temperature']:.6f}")
    print(f"Validation prefix NLL:      {prefix['macro_nll']:.6f}")
    print(f"Validation accuracy:        {prefix['macro_accuracy'] * 100:.2f}%")
    print(f"Config:                     {config_path}")
    print(f"Report:                     {report_path}")
    if sidecar:
        print(f"Model sidecar:              {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

