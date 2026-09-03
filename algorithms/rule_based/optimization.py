"""Fit and optimize additive rule weights from hard-label touch trials."""

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
from algorithms.rule_based.algorithm import RuleBasedClassifier
from algorithms.rule_based.config import RuleBasedConfig, load_config
from algorithms.rule_based.factory import DEFAULT_CONFIG_PATH


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


def _positive_float_list(value: str) -> tuple[float, ...]:
    parsed = _float_list(value)
    if any(item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("Values must be positive")
    return parsed


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _fit_rules(trials, config, smoothing: float) -> tuple[dict, dict]:
    shape_counts = {shape: 0 for shape in config.shapes}
    present = {
        shape: {feature: 0 for feature in config.features}
        for shape in config.shapes
    }
    for trial in trials:
        shape_counts[trial["shape"]] += 1
        observed = set(trial["sequence"])
        for feature in observed:
            present[trial["shape"]][feature] += 1

    total_trials = len(trials)
    rules = {}
    diagnostics = {}
    for shape in config.shapes:
        shape_total = shape_counts[shape]
        outside_total = total_trials - shape_total
        positive = {}
        negative = {}
        diagnostics[shape] = {}
        for feature in config.features:
            inside_hits = present[shape][feature]
            outside_hits = sum(
                present[other][feature]
                for other in config.shapes
                if other != shape
            )
            inside_probability = (inside_hits + smoothing) / (
                shape_total + 2.0 * smoothing
            )
            outside_probability = (outside_hits + smoothing) / (
                outside_total + 2.0 * smoothing
            )
            evidence = _logit(inside_probability) - _logit(outside_probability)
            positive[feature] = max(0.0, evidence)
            negative[feature] = max(0.0, -evidence)
            diagnostics[shape][feature] = {
                "inside_presence": inside_probability,
                "outside_presence": outside_probability,
                "signed_log_odds": evidence,
            }
        rules[shape] = {"positive": positive, "negative": negative}
    return rules, diagnostics


def _class_prior(trials, config, smoothing: float) -> dict:
    counts = {shape: smoothing for shape in config.shapes}
    for trial in trials:
        counts[trial["shape"]] += 1.0
    total = sum(counts.values())
    return {shape: counts[shape] / total for shape in config.shapes}


def _mapping(
    config,
    *,
    name: str,
    rules,
    class_prior,
    positive_weight: float,
    negative_weight: float,
    repetition_weight: float,
    temperature: float,
) -> dict:
    return {
        "schema_version": 1,
        "name": name,
        "features": list(config.features),
        "shapes": list(config.shapes),
        "aliases": dict(config.aliases),
        "rules": deepcopy(rules),
        "class_prior": dict(class_prior),
        "parameters": {
            "positive_weight": positive_weight,
            "negative_weight": negative_weight,
            "repetition_weight": repetition_weight,
            "score_temperature": temperature,
        },
    }


def optimize(
    train,
    validation,
    base,
    *,
    smoothing_values,
    repetition_values,
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
    candidates = []
    for smoothing in smoothing_values:
        rules, diagnostics = _fit_rules(parsed_train, base, smoothing)
        prior = _class_prior(parsed_train, base, smoothing)
        for repetition_weight in repetition_values:

            def objective(log_temperature):
                mapping = _mapping(
                    base,
                    name=name,
                    rules=rules,
                    class_prior=prior,
                    positive_weight=1.0,
                    negative_weight=1.0,
                    repetition_weight=repetition_weight,
                    temperature=math.exp(log_temperature),
                )
                config = RuleBasedConfig.from_mapping(mapping)
                metrics = evaluate_prefixes(
                    runs,
                    shapes=config.shapes,
                    create_classifier=lambda: RuleBasedClassifier(config),
                )
                return metrics["prefix"]["macro_nll"]

            result = minimize_scalar(
                objective,
                bounds=(math.log(0.25), math.log(6.0)),
                method="bounded",
                options={"xatol": 1e-5},
            )
            temperature = math.exp(result.x)
            mapping = _mapping(
                base,
                name=name,
                rules=rules,
                class_prior=prior,
                positive_weight=1.0,
                negative_weight=1.0,
                repetition_weight=repetition_weight,
                temperature=temperature,
            )
            config = RuleBasedConfig.from_mapping(mapping)
            metrics = evaluate_prefixes(
                runs,
                shapes=config.shapes,
                create_classifier=lambda: RuleBasedClassifier(config),
            )
            candidates.append(
                {
                    "key": (
                        metrics["prefix"]["macro_nll"],
                        metrics["prefix"]["macro_brier"],
                        -metrics["prefix"]["macro_accuracy"],
                    ),
                    "smoothing": smoothing,
                    "repetition_weight": repetition_weight,
                    "temperature": temperature,
                    "mapping": mapping,
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "temperature_optimizer_success": bool(result.success),
                }
            )
    best = min(candidates, key=lambda candidate: candidate["key"])
    report = {
        "method": "additive_feature_presence_rules",
        "selection_metric": "class-balanced mean validation prefix NLL",
        "training_trial_count": len(parsed_train),
        "validation_trial_count": len(parsed_validation),
        "validation_permutations": permutations,
        "seed": seed,
        "smoothing_candidates": list(smoothing_values),
        "repetition_weight_candidates": list(repetition_values),
        "selected_smoothing": best["smoothing"],
        "selected_repetition_weight": best["repetition_weight"],
        "selected_score_temperature": best["temperature"],
        "validation_metrics": best["metrics"],
        "rule_diagnostics": best["diagnostics"],
        "candidate_summary": [
            {
                "smoothing": candidate["smoothing"],
                "repetition_weight": candidate["repetition_weight"],
                "temperature": candidate["temperature"],
                "prefix_metrics": candidate["metrics"]["prefix"],
            }
            for candidate in candidates
        ],
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
    parser.add_argument("--smoothing", type=_positive_float_list, default=(0.05, 0.1, 0.5, 1.0))
    parser.add_argument("--repetition-weight", type=_float_list, default=(0.0, 0.1, 0.25, 0.5))
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
        repetition_values=args.repetition_weight,
        permutations=args.permutations,
        seed=args.seed,
        name=f"{base.name}_optimized_{run_timestamp}",
    )
    run_dir = args.output_dir.expanduser().resolve() / run_timestamp
    config_path = write_json(run_dir / f"rule_based_config_{run_timestamp}.json", optimized)
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
        config = RuleBasedConfig.from_mapping(optimized, source_path=config_path)
        report["test_metrics"] = evaluate_prefixes(
            runs,
            shapes=config.shapes,
            create_classifier=lambda: RuleBasedClassifier(config),
        )
        write_json(run_dir / f"test_{run_timestamp}.json", test)

    report["artifacts"] = {"config": str(config_path), "run_directory": str(run_dir)}
    report_path = write_json(run_dir / f"optimization_report_{run_timestamp}.json", report)
    sidecar = install_model_sidecar(config_path, args.model, ".rules.json") if args.model else None

    prefix = report["validation_metrics"]["prefix"]
    print(f"Selected smoothing:    {report['selected_smoothing']}")
    print(f"Repetition weight:     {report['selected_repetition_weight']}")
    print(f"Score temperature:     {report['selected_score_temperature']:.6f}")
    print(f"Validation prefix NLL: {prefix['macro_nll']:.6f}")
    print(f"Validation accuracy:   {prefix['macro_accuracy'] * 100:.2f}%")
    print(f"Config:                {config_path}")
    print(f"Report:                {report_path}")
    if sidecar:
        print(f"Model sidecar:         {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

