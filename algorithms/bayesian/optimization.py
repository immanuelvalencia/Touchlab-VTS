"""Training and validation selection for sequential Naive Bayes."""

from __future__ import annotations

from copy import deepcopy

from algorithms.hard_label_optimization import (
    evaluate_prefixes,
    normalize_trials,
    ordered_runs,
)

from .algorithm import BayesianShapePredictor


def _configuration(features, shapes, counts, smoothing):
    values = {}
    for feature in features:
        values[feature] = {}
        for shape in shapes:
            total = sum(counts[shape].values()) + smoothing * len(features)
            values[feature][shape] = (counts[shape][feature] + smoothing) / total
    return values


def optimize_bayesian(
    train,
    validation,
    *,
    features,
    shapes,
    aliases,
    smoothing_values=(0.01, 0.05, 0.1, 0.5, 1.0),
    permutations=5,
    seed=42,
):
    parsed_train = normalize_trials(
        train, features=features, shapes=shapes, aliases=aliases, name="Training"
    )
    parsed_validation = normalize_trials(
        validation,
        features=features,
        shapes=shapes,
        aliases=aliases,
        name="Validation",
    )
    counts = {
        shape: {feature: 0 for feature in features}
        for shape in shapes
    }
    for trial in parsed_train:
        for feature in trial["sequence"]:
            counts[trial["shape"]][feature] += 1
    runs = ordered_runs(parsed_validation, permutations=permutations, seed=seed)
    candidates = []
    for smoothing in smoothing_values:
        config = _configuration(features, shapes, counts, smoothing)

        def create():
            predictor = BayesianShapePredictor.__new__(BayesianShapePredictor)
            predictor.config_path = ""
            predictor.SHAPE_LIKELIHOODS = deepcopy(config)
            predictor.reset()
            return predictor

        metrics = evaluate_prefixes(runs, shapes=shapes, create_classifier=create)
        candidates.append(
            {
                "smoothing": smoothing,
                "config": config,
                "metrics": metrics,
                "key": (
                    metrics["prefix"]["macro_nll"],
                    metrics["prefix"]["macro_brier"],
                    -metrics["prefix"]["macro_accuracy"],
                ),
            }
        )
    best = min(candidates, key=lambda item: item["key"])
    return best["config"], {
        "method": "sequential_naive_bayesian",
        "training_trial_count": len(parsed_train),
        "validation_trial_count": len(parsed_validation),
        "validation_permutations": permutations,
        "seed": seed,
        "selection_metric": "class-balanced mean validation prefix NLL",
        "smoothing_candidates": list(smoothing_values),
        "selected_smoothing": best["smoothing"],
        "validation_metrics": best["metrics"],
        "candidate_summary": [
            {
                "smoothing": item["smoothing"],
                "prefix_metrics": item["metrics"]["prefix"],
            }
            for item in candidates
        ],
    }
