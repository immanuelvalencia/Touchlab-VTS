"""Optimize touch sequences for Modified Naive Bayes shape confidence.

The optimizer uses the same local-feature vocabulary as the simulator, but it
searches feature orders instead of sampling them. The objective is lexicographic:
first minimize the number of touches needed to reach a confidence threshold for
the target shape, then maximize confidence and margin at that touch count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.modified_naive_bayes.algorithm import ModifiedNaiveBayesClassifier
from algorithms.modified_naive_bayes.config import ModifiedNaiveBayesConfig, load_config


DEFAULT_CONFIG = PROJECT_ROOT / "algorithms" / "modified_naive_bayes" / "config.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "modified_nb_touch_optimization"
SHAPE_COLORS = {
    "cone": "#dc2626",
    "cube": "#2563eb",
    "cylinder": "#059669",
    "sphere": "#7c3aed",
    "square_pyramid": "#f59e0b",
}


@dataclass(frozen=True)
class TouchState:
    sequence: tuple[str, ...]
    prediction: str
    target_confidence: float
    global_confidence: float
    margin: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class OptimizedPlan:
    shape: str
    features: tuple[str, ...]
    sequence: tuple[str, ...]
    reached_threshold: bool
    threshold_touch: int | None
    final_prediction: str
    final_target_confidence: float
    final_global_confidence: float
    final_margin: float
    curve: tuple[TouchState, ...]


def display_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def slug(value: str) -> str:
    cleaned = []
    previous_separator = False
    for character in str(value).strip().lower():
        if character.isalnum():
            cleaned.append(character)
            previous_separator = False
        elif not previous_separator:
            cleaned.append("_")
            previous_separator = True
    return "".join(cleaned).strip("_") or "value"


def expected_features(config: ModifiedNaiveBayesConfig, shape: str) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in config.features
        if float(config.presence[shape][feature]) > 0.0
    )


def evaluate_sequence(
    config: ModifiedNaiveBayesConfig,
    shape: str,
    sequence: Sequence[str],
) -> tuple[TouchState, ...]:
    classifier = ModifiedNaiveBayesClassifier(config)
    curve = []
    for index, feature in enumerate(sequence, start=1):
        probabilities = classifier.update(feature)
        prediction = max(probabilities, key=probabilities.get)
        sorted_values = sorted(probabilities.values(), reverse=True)
        global_confidence = probabilities[prediction]
        runner_up = sorted_values[1] if len(sorted_values) > 1 else 0.0
        curve.append(
            TouchState(
                sequence=tuple(sequence[:index]),
                prediction=prediction,
                target_confidence=float(probabilities[shape]),
                global_confidence=float(global_confidence),
                margin=float(global_confidence - runner_up),
                probabilities={name: float(value) for name, value in probabilities.items()},
            )
        )
    return tuple(curve)


def state_key(
    state: TouchState,
    shape: str,
    *,
    threshold: float,
) -> tuple[float, ...]:
    correct = 1.0 if state.prediction == shape else 0.0
    reached = 1.0 if correct and state.target_confidence >= threshold else 0.0
    diversity = len(set(state.sequence)) / max(1, len(state.sequence))
    return (
        reached,
        correct,
        state.target_confidence,
        state.margin,
        diversity,
        -float(len(state.sequence)),
    )


def prune_candidates(
    candidates: Iterable[TouchState],
    shape: str,
    *,
    threshold: float,
    beam_width: int,
) -> list[TouchState]:
    best_by_signature = {}
    for candidate in candidates:
        counts = tuple(
            (feature, candidate.sequence.count(feature))
            for feature in sorted(set(candidate.sequence))
        )
        current = best_by_signature.get(counts)
        if current is None or state_key(candidate, shape, threshold=threshold) > state_key(
            current,
            shape,
            threshold=threshold,
        ):
            best_by_signature[counts] = candidate
    ordered = sorted(
        best_by_signature.values(),
        key=lambda state: state_key(state, shape, threshold=threshold),
        reverse=True,
    )
    return ordered[:beam_width]


def optimize_shape(
    config: ModifiedNaiveBayesConfig,
    shape: str,
    *,
    candidate_features: Sequence[str],
    threshold: float,
    max_touches: int,
    beam_width: int,
    prefer_unique_until_covered: bool,
) -> OptimizedPlan:
    beam: list[TouchState] = []
    best_overall: TouchState | None = None
    threshold_state: TouchState | None = None

    for depth in range(1, max_touches + 1):
        if depth == 1:
            sequences = [(feature,) for feature in candidate_features]
        else:
            sequences = []
            for state in beam:
                next_features = tuple(candidate_features)
                if prefer_unique_until_covered:
                    unused = tuple(
                        feature
                        for feature in candidate_features
                        if feature not in state.sequence
                    )
                    if unused:
                        next_features = unused
                for feature in next_features:
                    sequences.append((*state.sequence, feature))

        candidates = [
            evaluate_sequence(config, shape, sequence)[-1]
            for sequence in sequences
        ]
        beam = prune_candidates(
            candidates,
            shape,
            threshold=threshold,
            beam_width=beam_width,
        )
        best_at_depth = beam[0]
        if best_overall is None or state_key(best_at_depth, shape, threshold=threshold) > state_key(
            best_overall,
            shape,
            threshold=threshold,
        ):
            best_overall = best_at_depth

        reached = [
            state
            for state in beam
            if state.prediction == shape and state.target_confidence >= threshold
        ]
        if reached:
            threshold_state = max(
                reached,
                key=lambda state: (
                    state.target_confidence,
                    state.margin,
                    len(set(state.sequence)),
                ),
            )
            break

    selected = threshold_state or best_overall
    if selected is None:
        raise RuntimeError(f"No candidate sequence generated for {shape}")
    curve = evaluate_sequence(config, shape, selected.sequence)
    final = curve[-1]
    return OptimizedPlan(
        shape=shape,
        features=tuple(candidate_features),
        sequence=selected.sequence,
        reached_threshold=threshold_state is not None,
        threshold_touch=len(selected.sequence) if threshold_state is not None else None,
        final_prediction=final.prediction,
        final_target_confidence=final.target_confidence,
        final_global_confidence=final.global_confidence,
        final_margin=final.margin,
        curve=curve,
    )


def optimize_all(
    config: ModifiedNaiveBayesConfig,
    *,
    shapes: Sequence[str],
    threshold: float,
    max_touches: int,
    beam_width: int,
    prefer_unique_until_covered: bool,
) -> list[OptimizedPlan]:
    plans = []
    for shape in shapes:
        features = expected_features(config, shape)
        if not features:
            raise ValueError(f"No expected features configured for {shape}")
        plans.append(
            optimize_shape(
                config,
                shape,
                candidate_features=features,
                threshold=threshold,
                max_touches=max_touches,
                beam_width=beam_width,
                prefer_unique_until_covered=prefer_unique_until_covered,
            )
        )
    return plans


def write_summary(path: Path, plans: Sequence[OptimizedPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "shape",
                "candidate_features",
                "optimized_sequence",
                "reached_threshold",
                "threshold_touch",
                "final_prediction",
                "final_target_shape_confidence",
                "final_global_shape_confidence",
                "final_margin",
            ),
        )
        writer.writeheader()
        for plan in plans:
            writer.writerow(
                {
                    "shape": plan.shape,
                    "candidate_features": ";".join(plan.features),
                    "optimized_sequence": ";".join(plan.sequence),
                    "reached_threshold": plan.reached_threshold,
                    "threshold_touch": plan.threshold_touch or "",
                    "final_prediction": plan.final_prediction,
                    "final_target_shape_confidence": f"{plan.final_target_confidence * 100:.2f}",
                    "final_global_shape_confidence": f"{plan.final_global_confidence * 100:.2f}",
                    "final_margin": f"{plan.final_margin * 100:.2f}",
                }
            )


def write_touch_rows(path: Path, plans: Sequence[OptimizedPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "shape",
                "touch",
                "feature",
                "prediction",
                "target_shape_confidence",
                "global_shape_confidence",
                "margin",
            ),
        )
        writer.writeheader()
        for plan in plans:
            for touch, state in enumerate(plan.curve, start=1):
                writer.writerow(
                    {
                        "shape": plan.shape,
                        "touch": touch,
                        "feature": state.sequence[-1],
                        "prediction": state.prediction,
                        "target_shape_confidence": f"{state.target_confidence * 100:.2f}",
                        "global_shape_confidence": f"{state.global_confidence * 100:.2f}",
                        "margin": f"{state.margin * 100:.2f}",
                    }
                )


def write_json_report(path: Path, config_path: Path, plans: Sequence[OptimizedPlan], args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(config_path),
        "objective": {
            "primary": "minimize touches to threshold",
            "secondary": "maximize target confidence and posterior margin",
            "threshold": args.threshold,
            "max_touches": args.max_touches,
            "beam_width": args.beam_width,
            "prefer_unique_until_covered": not args.allow_early_repeats,
        },
        "plans": [
            {
                "shape": plan.shape,
                "candidate_features": list(plan.features),
                "optimized_sequence": list(plan.sequence),
                "reached_threshold": plan.reached_threshold,
                "threshold_touch": plan.threshold_touch,
                "final_prediction": plan.final_prediction,
                "final_target_shape_confidence": plan.final_target_confidence,
                "final_global_shape_confidence": plan.final_global_confidence,
                "final_margin": plan.final_margin,
                "curve": [
                    {
                        "touch": index,
                        "feature": state.sequence[-1],
                        "prediction": state.prediction,
                        "target_shape_confidence": state.target_confidence,
                        "global_shape_confidence": state.global_confidence,
                        "margin": state.margin,
                        "probabilities": state.probabilities,
                    }
                    for index, state in enumerate(plan.curve, start=1)
                ],
            }
            for plan in plans
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def plot_plans(path: Path, plans: Sequence[OptimizedPlan], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    max_touch = 1
    for plan in plans:
        touches = list(range(1, len(plan.curve) + 1))
        max_touch = max(max_touch, len(touches))
        confidence = [state.target_confidence * 100 for state in plan.curve]
        axis.plot(
            touches,
            confidence,
            color=SHAPE_COLORS.get(plan.shape, "#374151"),
            linewidth=2.1,
            marker="o",
            markersize=3.2,
            label=display_label(plan.shape),
        )
    axis.axhline(50, color="#9ca3af", linestyle=":", linewidth=1.0, alpha=0.9)
    axis.axhline(threshold * 100, color="#4b5563", linestyle="--", linewidth=1.0, alpha=0.85)
    axis.set_title("Optimized Modified Naive Bayes Touch Plans")
    axis.set_xlabel("Touch")
    axis.set_ylabel("Target-shape confidence (%)")
    axis.set_xlim(1, max_touch)
    axis.set_ylim(0, 105)
    axis.set_xticks(list(range(1, max_touch + 1)))
    y_ticks = sorted({0, 20, 40, 50, 60, 80, int(round(threshold * 100)), 100})
    axis.set_yticks(y_ticks)
    axis.grid(True, linestyle=":", linewidth=0.75, alpha=0.65)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_shapes(value: str | None, config: ModifiedNaiveBayesConfig) -> tuple[str, ...]:
    if not value:
        return tuple(config.shapes)
    shapes = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(shapes) - set(config.shapes))
    if unknown:
        raise ValueError("Unknown shapes: " + ", ".join(unknown))
    return shapes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shapes", help="Comma-separated shapes. Defaults to all shapes.")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--max-touches", type=int, default=30)
    parser.add_argument("--beam-width", type=int, default=250)
    parser.add_argument(
        "--allow-early-repeats",
        action="store_true",
        help="Allow repeated features before every expected feature has appeared once.",
    )
    parser.add_argument("--timestamp", action="store_true", help="Write outputs under a timestamped subfolder.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.max_touches < 1:
        raise ValueError("--max-touches must be positive")
    if args.beam_width < 1:
        raise ValueError("--beam-width must be positive")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    shapes = parse_shapes(args.shapes, config)
    output_dir = args.output_dir.expanduser().resolve()
    if args.timestamp:
        output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    plans = optimize_all(
        config,
        shapes=shapes,
        threshold=args.threshold,
        max_touches=args.max_touches,
        beam_width=args.beam_width,
        prefer_unique_until_covered=not args.allow_early_repeats,
    )

    summary_path = output_dir / "modified_nb_touch_plan_summary.csv"
    touches_path = output_dir / "modified_nb_touch_plan_touches.csv"
    report_path = output_dir / "modified_nb_touch_plan_report.json"
    chart_path = output_dir / "modified_nb_touch_plan_confidence.png"
    write_summary(summary_path, plans)
    write_touch_rows(touches_path, plans)
    write_json_report(report_path, config_path, plans, args)
    plot_plans(chart_path, plans, args.threshold)

    print("Optimized Modified Naive Bayes touch plans")
    for plan in plans:
        threshold_text = (
            f"touch {plan.threshold_touch}"
            if plan.threshold_touch is not None
            else f">{args.max_touches} touches"
        )
        print(
            f"- {display_label(plan.shape)}: {threshold_text}, "
            f"final {plan.final_prediction} "
            f"({plan.final_global_confidence * 100:.1f}%), "
            f"sequence: {' -> '.join(plan.sequence)}"
        )
    print(f"Summary: {summary_path}")
    print(f"Touches: {touches_path}")
    print(f"Report:  {report_path}")
    print(f"Chart:   {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
