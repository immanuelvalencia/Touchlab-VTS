"""Fit and optimize Dirichlet-Multinomial models from hard-label trials."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
import sys

from scipy.optimize import minimize, minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.dirichlet_multinomial.algorithm import DirichletMultinomialClassifier
from algorithms.dirichlet_multinomial.config import DirichletMultinomialConfig, load_config
from algorithms.dirichlet_multinomial.factory import DEFAULT_CONFIG_PATH
from algorithms.hard_label_optimization import (
    evaluate_prefixes,
    install_model_sidecar,
    load_trials,
    normalize_trials,
    ordered_runs,
    timestamp,
    write_json,
)


DEFAULT_TRAIN = PROJECT_ROOT / "trials" / "rfs_train.json"
DEFAULT_VALIDATION = PROJECT_ROOT / "trials" / "rfs_validation.json"
DEFAULT_RUNS = Path(__file__).resolve().parent / "runs"


def _count_vectors(trials, features, shapes):
    vectors = defaultdict(list)
    for trial in trials:
        counts = {feature: 0 for feature in features}
        for feature in trial["sequence"]:
            counts[feature] += 1
        vectors[trial["shape"]].append([counts[feature] for feature in features])
    return {shape: vectors[shape] for shape in shapes}


def _negative_log_likelihood(log_alpha, vectors) -> float:
    alpha = [math.exp(value) for value in log_alpha]
    alpha_zero = sum(alpha)
    total = 0.0
    for counts in vectors:
        touch_count = sum(counts)
        log_probability = math.lgamma(touch_count + 1.0) - sum(
            math.lgamma(count + 1.0) for count in counts
        )
        log_probability += math.lgamma(alpha_zero) - math.lgamma(
            alpha_zero + touch_count
        )
        log_probability += sum(
            math.lgamma(value + count) - math.lgamma(value)
            for value, count in zip(alpha, counts)
        )
        total -= log_probability
    return total


def _fit_alpha_templates(trials, config, initial_concentration, max_iterations):
    vectors = _count_vectors(trials, config.features, config.shapes)
    templates = {}
    fitting = {}
    for shape in config.shapes:
        totals = [0.5] * len(config.features)
        for counts in vectors[shape]:
            totals = [left + right for left, right in zip(totals, counts)]
        mass = sum(totals)
        initial = [
            math.log(max(1e-3, initial_concentration * value / mass))
            for value in totals
        ]
        result = minimize(
            _negative_log_likelihood,
            initial,
            args=(vectors[shape],),
            method="L-BFGS-B",
            bounds=[(math.log(1e-3), math.log(1e4))] * len(config.features),
            options={"maxiter": max_iterations, "ftol": 1e-10},
        )
        alpha = [math.exp(value) for value in result.x]
        templates[shape] = {
            feature: alpha[index] for index, feature in enumerate(config.features)
        }
        fitting[shape] = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "negative_log_likelihood": float(result.fun),
            "concentration": sum(alpha),
            "reached_parameter_bound": any(
                value <= 1.001e-3 or value >= 0.999e4 for value in alpha
            ),
        }
    return templates, fitting


def _mapping(config, *, name, alpha_templates, temperature):
    return {
        "schema_version": 1,
        "name": name,
        "features": list(config.features),
        "shapes": list(config.shapes),
        "aliases": dict(config.aliases),
        "alpha_templates": deepcopy(alpha_templates),
        "class_prior": dict(config.class_prior),
        "parameters": {"score_temperature": temperature},
    }


def optimize(
    train,
    validation,
    base,
    *,
    permutations,
    seed,
    initial_concentration,
    max_iterations,
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
    alpha_templates, fitting = _fit_alpha_templates(
        parsed_train,
        base,
        initial_concentration,
        max_iterations,
    )
    runs = ordered_runs(parsed_validation, permutations=permutations, seed=seed)

    def objective(log_temperature):
        mapping = _mapping(
            base,
            name=name,
            alpha_templates=alpha_templates,
            temperature=math.exp(log_temperature),
        )
        config = DirichletMultinomialConfig.from_mapping(mapping)
        metrics = evaluate_prefixes(
            runs,
            shapes=config.shapes,
            create_classifier=lambda: DirichletMultinomialClassifier(config),
        )
        return metrics["prefix"]["macro_nll"]

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.25), math.log(4.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    temperature = math.exp(result.x)
    mapping = _mapping(
        base,
        name=name,
        alpha_templates=alpha_templates,
        temperature=temperature,
    )
    config = DirichletMultinomialConfig.from_mapping(mapping)
    metrics = evaluate_prefixes(
        runs,
        shapes=config.shapes,
        create_classifier=lambda: DirichletMultinomialClassifier(config),
    )
    report = {
        "method": "dirichlet_multinomial_empirical_bayes",
        "selection_metric": "class-balanced mean validation prefix NLL",
        "training_trial_count": len(parsed_train),
        "validation_trial_count": len(parsed_validation),
        "validation_permutations": permutations,
        "seed": seed,
        "alpha_estimation": {
            "method": "per-class maximum marginal likelihood in log-alpha space",
            "initial_concentration": initial_concentration,
            "bounds": [1e-3, 1e4],
            "max_iterations": max_iterations,
            "per_shape": fitting,
        },
        "selected_score_temperature": temperature,
        "temperature_optimizer_success": bool(result.success),
        "temperature_at_search_boundary": (
            temperature <= 0.251 or temperature >= 3.999
        ),
        "validation_metrics": metrics,
    }
    return mapping, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--test-permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-concentration", type=float, default=10.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.initial_concentration <= 0.0:
        raise ValueError("--initial-concentration must be positive")
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be positive")
    base = load_config(args.base_config)
    train = load_trials(args.train)
    validation = load_trials(args.validation)
    run_timestamp = timestamp()
    optimized, report = optimize(
        train,
        validation,
        base,
        permutations=args.permutations,
        seed=args.seed,
        initial_concentration=args.initial_concentration,
        max_iterations=args.max_iterations,
        name=f"{base.name}_optimized_{run_timestamp}",
    )
    run_dir = args.output_dir.expanduser().resolve() / run_timestamp
    config_path = write_json(
        run_dir / f"dirichlet_multinomial_config_{run_timestamp}.json", optimized
    )
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
        config = DirichletMultinomialConfig.from_mapping(optimized, source_path=config_path)
        report["test_metrics"] = evaluate_prefixes(
            runs,
            shapes=config.shapes,
            create_classifier=lambda: DirichletMultinomialClassifier(config),
        )
        write_json(run_dir / f"test_{run_timestamp}.json", test)

    report["artifacts"] = {"config": str(config_path), "run_directory": str(run_dir)}
    report_path = write_json(run_dir / f"optimization_report_{run_timestamp}.json", report)
    sidecar = install_model_sidecar(config_path, args.model, ".dm.json") if args.model else None

    prefix = report["validation_metrics"]["prefix"]
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
