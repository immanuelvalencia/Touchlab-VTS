"""Consolidate shape-simulation CSV exports into confidence line graphs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "output" / "shape_simulations"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "consolidated_shape_confidence.png"
DEFAULT_SUMMARY = DEFAULT_INPUT_DIR / "consolidated_shape_confidence.csv"
DEFAULT_SPLIT_PREFIX = DEFAULT_INPUT_DIR / "consolidated_shape_confidence"

FILENAME_RE = re.compile(
    r"^shape_sim_(?P<shape>.+)_(?P<order>in_order|random|random_unique_first)_"
    r"(?P<trials>\d+)trials_(?P<touches>\d+)touches_.*\.csv$"
)
CONFIDENCE_RE = re.compile(r"\((?P<confidence>\d+(?:\.\d+)?)%\)")

SHAPE_COLORS = {
    "cone": "#dc2626",
    "cube": "#2563eb",
    "cylinder": "#059669",
    "sphere": "#7c3aed",
    "square_pyramid": "#f59e0b",
}

ORDER_STYLES = {
    "in_order": "-",
    "random": "--",
    "random_unique_first": ":",
}
ALGORITHM_STYLES = ["-", "--", ":", "-."]
ALGORITHM_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]


def display_label(value: str) -> str:
    return value.replace("_", " ").title()


def parse_filename(path: Path) -> tuple[str, str]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot infer shape/order from filename: {path.name}")
    return match.group("shape"), match.group("order")


def parse_confidence(value: str) -> float:
    match = CONFIDENCE_RE.search(value)
    if not match:
        raise ValueError(f"Cannot parse confidence value: {value!r}")
    return float(match.group("confidence"))


def algorithm_from_confidence_column(column: str) -> str:
    if column == "global_shape_confidence":
        return "Global Shape"
    suffix = "_global_shape_confidence"
    if not column.endswith(suffix):
        return column
    return display_label(column[: -len(suffix)])


def read_csv(path: Path) -> list[dict[str, object]]:
    shape, order = parse_filename(path)
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"trial", "touch", "feature"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        confidence_columns = sorted(
            column for column in fieldnames if column.endswith("_global_shape_confidence")
        )
        if "global_shape_confidence" in fieldnames:
            confidence_columns.insert(0, "global_shape_confidence")
        if not confidence_columns:
            raise ValueError(f"{path.name} has no algorithm confidence columns")
        for row in reader:
            for column in confidence_columns:
                confidence = str(row[column]).strip()
                if not confidence:
                    continue
                rows.append(
                    {
                        "shape": shape,
                        "order": order,
                        "algorithm": (
                            str(row.get("algorithm") or "").strip()
                            if column == "global_shape_confidence"
                            else ""
                        )
                        or algorithm_from_confidence_column(column),
                        "trial": int(row["trial"]),
                        "touch": int(row["touch"]),
                        "feature": row["feature"],
                        "confidence": parse_confidence(confidence),
                    }
                )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["shape"], row["order"], row["algorithm"], row["touch"])].append(
            float(row["confidence"])
        )

    summary = []
    for (shape, order, algorithm, touch), values in sorted(grouped.items()):
        summary.append(
            {
                "shape": shape,
                "order": order,
                "algorithm": algorithm,
                "touch": touch,
                "mean_confidence": sum(values) / len(values),
                "trials": len(values),
            }
        )
    return summary


def write_summary(path: Path, summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("shape", "order", "algorithm", "touch", "mean_confidence", "trials"),
        )
        writer.writeheader()
        writer.writerows(summary)


def plot_summary(
    path: Path,
    summary: list[dict[str, object]],
    order_filter: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series = defaultdict(list)
    for row in summary:
        if order_filter is not None and row["order"] != order_filter:
            continue
        series[(row["shape"], row["order"], row["algorithm"])].append(row)
    if not series:
        return
    max_touch = max(
        int(row["touch"])
        for rows in series.values()
        for row in rows
    )

    plt.figure(figsize=(9.0, 5.2))
    algorithms = sorted({algorithm for _shape, _order, algorithm in series})
    orders = sorted({order for _shape, order, _algorithm in series})
    for (shape, order, algorithm), rows in sorted(series.items()):
        rows = sorted(rows, key=lambda row: int(row["touch"]))
        touches = [int(row["touch"]) for row in rows]
        confidence = [float(row["mean_confidence"]) for row in rows]
        style_index = algorithms.index(algorithm) if algorithm in algorithms else 0
        label = f"{display_label(shape)} - {algorithm}"
        if len(orders) > 1:
            label += f" ({order.replace('_', ' ')})"
        plt.plot(
            touches,
            confidence,
            linestyle=ALGORITHM_STYLES[style_index % len(ALGORITHM_STYLES)],
            marker=ALGORITHM_MARKERS[style_index % len(ALGORITHM_MARKERS)],
            color=SHAPE_COLORS.get(shape, "#374151"),
            linewidth=2.0,
            markersize=2.6,
            label=label,
        )

    plt.axhline(50, color="#9ca3af", linestyle=":", linewidth=1.1, alpha=0.9)
    plt.axhline(90, color="#4b5563", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.xlabel("Touch")
    plt.ylabel("Global shape confidence (%)")
    title = "Shape Confidence Across Simulated Touches"
    if order_filter is not None:
        title += f" ({order_filter.replace('_', ' ').title()})"
    plt.title(title)
    plt.xlim(1, max_touch)
    x_ticks = list(range(3, max_touch + 1, 3))
    plt.xticks(x_ticks or [max_touch])
    plt.ylim(0, 105)
    plt.yticks([0, 20, 40, 50, 60, 80, 90, 100])
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.65)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="*", type=Path)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split-prefix", type=Path, default=DEFAULT_SPLIT_PREFIX)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_files = args.csv_files or sorted(args.input_dir.glob("shape_sim_*.csv"))
    if not csv_files:
        raise SystemExit(f"No shape simulation CSV files found in {args.input_dir}")

    rows = []
    for path in csv_files:
        rows.extend(read_csv(path))
    summary = summarize(rows)
    write_summary(args.summary, summary)
    plot_summary(args.output, summary)
    split_outputs = []
    for order in sorted({str(row["order"]) for row in summary}):
        output_path = args.split_prefix.with_name(f"{args.split_prefix.name}_{order}.png")
        plot_summary(output_path, summary, order_filter=order)
        split_outputs.append(output_path)

    print(f"Read {len(csv_files)} CSV files")
    print(f"Summary: {args.summary}")
    print(f"Plot: {args.output}")
    for output_path in split_outputs:
        print(f"Split plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
