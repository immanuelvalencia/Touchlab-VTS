"""Simulate rule-based primitive-shape confidence from unique tactile features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rule_based.algorithm import RuleBasedClassifier
from algorithms.rule_based.config import RuleBasedConfig


DEFAULT_CONFIG = PROJECT_ROOT / "algorithms" / "rule_based" / "config.json"
DEFAULT_DATASET = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "publication"
    / "Local_Geometric_Feature_Learning_from_Handheld_Visuo-Tactile_Contacts"
    / "figures"
)

SHAPE_ALIASES = {
    "pyramid": "square_pyramid",
}

TOUCH_POLICY = {
    "cone": {"min": 3, "max": 4},
    "cube": {"min": 3, "max": 4},
    "cylinder": {"min": 3, "max": 4},
    "sphere": {"min": 2, "max": 3},
    "square_pyramid": {"min": 3, "max": 4},
}

COLORS = {
    "cone": "#dc2626",
    "cube": "#2563eb",
    "cylinder": "#059669",
    "sphere": "#7c3aed",
    "square_pyramid": "#f59e0b",
}


def load_rule_config(path: Path, score_temperature: float | None) -> RuleBasedConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if score_temperature is not None:
        raw["parameters"]["score_temperature"] = score_temperature
    return RuleBasedConfig.from_mapping(raw, source_path=path)


def canonical_shape(value: str) -> str:
    label = str(value).strip().lower()
    return SHAPE_ALIASES.get(label, label)


def canonical_feature(value: str, aliases: dict[str, str]) -> str:
    label = str(value).strip().lower()
    return aliases.get(label, label)


def scan_dataset_features(dataset_dir: Path, config: RuleBasedConfig) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    valid_shapes = set(config.shapes)
    valid_features = set(config.features)
    aliases = dict(config.aliases)

    for metadata_path in dataset_dir.rglob("*metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            continue

        shape = canonical_shape(metadata.get("label", ""))
        feature = canonical_feature(
            metadata.get("custom_fields", {}).get("local_feature", ""),
            aliases,
        )
        if shape in valid_shapes and feature in valid_features:
            counts[shape][feature] += 1
    return counts


def rule_feature_sequence(
    shape: str,
    config: RuleBasedConfig,
    dataset_counts: dict[str, Counter],
) -> list[dict[str, object]]:
    positive = config.rules[shape]["positive"]
    candidates = [
        feature
        for feature in config.features
        if float(positive[feature]) > 0.0
    ]
    candidates.sort(
        key=lambda feature: (
            -float(positive[feature]),
            -dataset_counts.get(shape, Counter()).get(feature, 0),
            feature,
        )
    )
    limit = TOUCH_POLICY[shape]["max"]
    return [
        {
            "feature": feature,
            "positive_weight": float(positive[feature]),
            "dataset_count": dataset_counts.get(shape, Counter()).get(feature, 0),
        }
        for feature in candidates[:limit]
    ]


def simulate_shape(
    shape: str,
    sequence: list[dict[str, object]],
    config: RuleBasedConfig,
    target_confidence: float,
) -> tuple[list[dict[str, object]], int | None]:
    classifier = RuleBasedClassifier(config)
    rows = []
    accepted_touch = None
    min_touches = TOUCH_POLICY[shape]["min"]

    for touch_index, item in enumerate(sequence, start=1):
        probabilities = classifier.update(str(item["feature"]))
        predicted_shape = max(probabilities, key=probabilities.get)
        predicted_confidence = probabilities[predicted_shape]
        true_confidence = probabilities[shape]
        if (
            accepted_touch is None
            and touch_index >= min_touches
            and predicted_shape == shape
            and predicted_confidence >= target_confidence
        ):
            accepted_touch = touch_index

        rows.append(
            {
                "shape": shape,
                "touch": touch_index,
                "feature": item["feature"],
                "dataset_count": item["dataset_count"],
                "positive_weight": item["positive_weight"],
                "true_shape_confidence": true_confidence,
                "predicted_shape": predicted_shape,
                "predicted_confidence": predicted_confidence,
                "accepted_touch": accepted_touch == touch_index,
            }
        )
    return rows, accepted_touch


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shape",
        "touch",
        "feature",
        "dataset_count",
        "positive_weight",
        "true_shape_confidence",
        "predicted_shape",
        "predicted_confidence",
        "accepted_touch",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    path: Path,
    rows: list[dict[str, object]],
    accepted: dict[str, int | None],
    target_confidence: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shape",
        "touch_range_to_threshold",
        "accepted_touch",
        "confidence_threshold",
        "final_confidence",
        "unique_features",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for shape in TOUCH_POLICY:
            shape_rows = [row for row in rows if row["shape"] == shape]
            accepted_touch = accepted[shape]
            touch_range = (
                f"{accepted_touch}-{TOUCH_POLICY[shape]['max']}"
                if accepted_touch is not None
                else "not reached"
            )
            writer.writerow(
                {
                    "shape": shape,
                    "touch_range_to_threshold": touch_range,
                    "accepted_touch": accepted_touch or "",
                    "confidence_threshold": target_confidence,
                    "final_confidence": shape_rows[-1]["true_shape_confidence"],
                    "unique_features": "; ".join(str(row["feature"]) for row in shape_rows),
                }
            )


def plot_confidence(path: Path, rows: list[dict[str, object]], target_confidence: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.2))

    for shape in TOUCH_POLICY:
        shape_rows = [row for row in rows if row["shape"] == shape]
        if not shape_rows:
            continue
        touches = [int(row["touch"]) for row in shape_rows]
        confidence = [float(row["true_shape_confidence"]) * 100.0 for row in shape_rows]
        plt.plot(
            touches,
            confidence,
            marker="o",
            linewidth=2,
            color=COLORS[shape],
            label=shape.replace("_", " ").title(),
        )

    plt.axhline(
        target_confidence * 100.0,
        color="#4b5563",
        linestyle="--",
        linewidth=1.2,
        label=f"{int(target_confidence * 100)}% threshold",
    )
    plt.xlabel("Number of touches")
    plt.ylabel("Confidence (%)")
    plt.xticks([1, 2, 3, 4])
    plt.ylim(0, 105)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-confidence", type=float, default=0.90)
    parser.add_argument(
        "--score-temperature",
        type=float,
        default=0.40,
        help="Confidence plotting temperature. The rule weights are unchanged.",
    )
    parser.add_argument(
        "--plot-name",
        default="rule_based_confidence_curves.png",
        help="Output PNG filename.",
    )
    parser.add_argument(
        "--csv-name",
        default="rule_based_confidence_curves.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--summary-name",
        default="rule_based_confidence_summary.csv",
        help="Output summary CSV filename.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_confidence <= 0.0 or args.target_confidence >= 1.0:
        raise ValueError("--target-confidence must be between 0 and 1")
    if args.score_temperature <= 0.0:
        raise ValueError("--score-temperature must be positive")

    config = load_rule_config(args.config, args.score_temperature)
    dataset_counts = scan_dataset_features(args.dataset_dir, config)

    rows = []
    accepted = {}
    for shape in TOUCH_POLICY:
        sequence = rule_feature_sequence(shape, config, dataset_counts)
        shape_rows, accepted_touch = simulate_shape(
            shape,
            sequence,
            config,
            args.target_confidence,
        )
        rows.extend(shape_rows)
        accepted[shape] = accepted_touch

    plot_path = args.output_dir / args.plot_name
    csv_path = args.output_dir / args.csv_name
    summary_path = args.output_dir / args.summary_name
    plot_confidence(plot_path, rows, args.target_confidence)
    write_csv(csv_path, rows)
    write_summary_csv(summary_path, rows, accepted, args.target_confidence)

    print(f"Plot: {plot_path}")
    print(f"CSV:  {csv_path}")
    print(f"Summary CSV: {summary_path}")
    print("Accepted touch counts:")
    for shape in TOUCH_POLICY:
        touch = accepted[shape]
        status = str(touch) if touch is not None else "not reached"
        print(f"  {shape}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
