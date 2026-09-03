"""Tkinter app for simulating multi-touch primitive-shape algorithms."""

from __future__ import annotations

import csv
import datetime
import os
import random
import re
import sys
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.registry import get_algorithm_specs


DEFAULT_FEATURES = (
    "planar",
    "single_curvature",
    "double_curvature",
    "straight_edge",
    "circular_rim",
    "multi_face_vertex",
    "sharp_apex",
)

DEFAULT_SHAPES = (
    "cube",
    "sphere",
    "cylinder",
    "cone",
    "square_pyramid",
)

FALLBACK_PRESETS = {
    "cube": {"planar", "straight_edge", "multi_face_vertex"},
    "sphere": {"double_curvature"},
    "cylinder": {"planar", "single_curvature", "circular_rim"},
    "cone": {"planar", "single_curvature", "circular_rim", "sharp_apex"},
    "square_pyramid": {"planar", "straight_edge", "multi_face_vertex", "sharp_apex"},
}

SIMULATION_FILENAME_RE = re.compile(
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
ALGORITHM_COLORS = {
    "Bayesian": "#64748b",
    "Naive Bayes": "#2563eb",
    "Modified Naive Bayes": "#dc2626",
    "Single Touch Baseline": "#f59e0b",
    "Rule-Based": "#059669",
    "Bag of Features": "#7c3aed",
    "Dirichlet-Multinomial": "#0891b2",
    "Set Evidence": "#be123c",
}
ALGORITHM_STYLES = ["-", "--", ":", "-."]
ALGORITHM_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]


def display_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def slug(value: str) -> str:
    cleaned = []
    previous_was_separator = False
    for character in str(value).strip().lower():
        if character.isalnum():
            cleaned.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            cleaned.append("_")
            previous_was_separator = True
    return "".join(cleaned).strip("_") or "value"


def confidence_column_name(algorithm_name: str, *, single_algorithm: bool = False) -> str:
    if single_algorithm:
        return "global_shape_confidence"
    return f"{slug(algorithm_name)}_global_shape_confidence"


def algorithm_name_from_confidence_column(column: str) -> str:
    if column == "global_shape_confidence":
        return "Global Shape"
    suffix = "_global_shape_confidence"
    if not column.endswith(suffix):
        return column
    stem = column[: -len(suffix)]
    names = {slug(spec.display_name): spec.display_name for spec in get_algorithm_specs()}
    return names.get(stem, display_label(stem))


class ShapeSimulationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TouchLab Shape Algorithm Simulator")
        self.root.geometry("1120x720")
        self.root.minsize(980, 620)

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", font=("Segoe UI", 10))
        self.style.configure("Treeview", rowheight=26)
        self.style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        self.specs = get_algorithm_specs()
        self.algorithm_vars: dict[str, tk.BooleanVar] = {}
        self.algorithms = {}
        self.features = list(DEFAULT_FEATURES)
        self.shapes = list(DEFAULT_SHAPES)
        self.feature_vars: dict[str, tk.BooleanVar] = {}
        self.history: list[dict[str, object]] = []
        self.current_trial = 1
        self.touch_counter = 0
        self.auto_queue: list[dict[str, object]] = []
        self.auto_running = False
        self.trial_count_var = tk.IntVar(value=1)
        self.touch_count_var = tk.IntVar(value=4)
        self.order_mode_var = tk.StringVar(value="In order")
        self.output_dir_var = tk.StringVar(
            value=str(PROJECT_ROOT / "output" / "shape_simulations")
        )

        self.load_algorithms()
        self.presets = self.build_presets()
        self.create_widgets()
        self.apply_shape_preset()
        self.reset_trial()

    def load_algorithms(self) -> None:
        self.algorithms = {}
        features = []
        shapes = []

        for spec in self.specs:
            try:
                algorithm = spec.create()
            except Exception as exc:
                print(f"Failed to load {spec.display_name}: {exc}")
                continue
            self.algorithms[spec.key] = (spec, algorithm)
            for feature in getattr(algorithm, "features", ()):
                if feature not in features:
                    features.append(feature)
            candidate_shapes = getattr(algorithm, "shapes", None)
            if candidate_shapes is None and hasattr(algorithm, "shape_probs"):
                candidate_shapes = tuple(algorithm.shape_probs)
            for shape in candidate_shapes or ():
                if shape not in shapes:
                    shapes.append(shape)

        if features:
            self.features = features
        if shapes:
            self.shapes = shapes

    def build_presets(self) -> dict[str, set[str]]:
        presets = {shape: set(FALLBACK_PRESETS.get(shape, set())) for shape in self.shapes}
        for _, algorithm in self.algorithms.values():
            config = getattr(algorithm, "config", None)
            rules = getattr(config, "rules", None)
            if not rules:
                continue
            for shape in self.shapes:
                positive = rules.get(shape, {}).get("positive", {})
                selected = {
                    feature
                    for feature, weight in positive.items()
                    if float(weight) > 0.0
                }
                if selected:
                    presets[shape] = selected
            break
        return presets

    def create_widgets(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(outer, text="Simulation Setup", padding=12)
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        results = ttk.Frame(outer)
        results.grid(row=0, column=1, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(3, weight=1)

        ttk.Label(controls, text="Algorithms", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        for spec in self.specs:
            enabled = spec.key in self.algorithms
            var = tk.BooleanVar(value=enabled)
            self.algorithm_vars[spec.key] = var
            ttk.Checkbutton(
                controls,
                text=spec.display_name,
                variable=var,
                command=self.on_algorithm_selection_changed,
                state=tk.NORMAL if enabled else tk.DISABLED,
            ).pack(anchor=tk.W, pady=1)

        ttk.Separator(controls).pack(fill=tk.X, pady=10)

        ttk.Label(controls, text="Primitive shape", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.shape_var = tk.StringVar(value=self.shapes[0] if self.shapes else "")
        shape_box = ttk.Combobox(
            controls,
            textvariable=self.shape_var,
            values=self.shapes,
            state="readonly",
            width=24,
        )
        shape_box.pack(fill=tk.X, pady=(2, 8))
        shape_box.bind("<<ComboboxSelected>>", lambda _event: self.apply_shape_preset())

        ttk.Label(controls, text="Simulated local features", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        feature_frame = ttk.Frame(controls)
        feature_frame.pack(fill=tk.X, pady=(2, 8))
        for feature in self.features:
            var = tk.BooleanVar(value=False)
            self.feature_vars[feature] = var
            ttk.Checkbutton(
                feature_frame,
                text=display_label(feature),
                variable=var,
            ).pack(anchor=tk.W, pady=1)

        preset_buttons = ttk.Frame(controls)
        preset_buttons.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(preset_buttons, text="Use Expected", command=self.apply_shape_preset).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(preset_buttons, text="Clear", command=self.clear_features).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        ttk.Label(controls, text="Trials").pack(anchor=tk.W)
        ttk.Spinbox(
            controls,
            from_=1,
            to=100,
            increment=1,
            textvariable=self.trial_count_var,
            width=10,
        ).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(controls, text="Touches to simulate").pack(anchor=tk.W)
        ttk.Spinbox(
            controls,
            from_=1,
            to=50,
            increment=1,
            textvariable=self.touch_count_var,
            width=10,
        ).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(controls, text="Touch order").pack(anchor=tk.W)
        order_box = ttk.Combobox(
            controls,
            textvariable=self.order_mode_var,
            values=("In order", "Random", "Random unique first"),
            state="readonly",
            width=20,
        )
        order_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(controls, text="Auto interval (ms)").pack(anchor=tk.W)
        self.interval_var = tk.IntVar(value=700)
        ttk.Spinbox(
            controls,
            from_=100,
            to=5000,
            increment=100,
            textvariable=self.interval_var,
            width=10,
        ).pack(fill=tk.X, pady=(0, 10))

        ttk.Button(controls, text="Feed Next Touch", command=self.feed_next_touch).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Run Simulation", command=self.start_auto_feed).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Stop Auto Feed", command=self.stop_auto_feed).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Reset Trial", command=self.reset_trial).pack(fill=tk.X, pady=(10, 2))
        ttk.Button(controls, text="Optimize Modified NB", command=self.optimize_modified_nb_plan).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Export CSV + Chart", command=self.export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Run All Shapes + Export", command=self.run_all_shapes_and_export).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Consolidate Graph", command=self.consolidate_all_graphs).pack(fill=tk.X, pady=(2, 8))

        ttk.Label(controls, text="CSV save folder").pack(anchor=tk.W)
        folder_row = ttk.Frame(controls)
        folder_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Entry(
            folder_row,
            textvariable=self.output_dir_var,
            width=24,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(folder_row, text="Browse", command=self.browse_output_dir).pack(
            side=tk.LEFT,
            padx=(6, 0),
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var, wraplength=250).pack(fill=tk.X, pady=(12, 0))

        action_bar = ttk.Frame(results)
        action_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        action_bar.columnconfigure(0, weight=1)
        action_bar.columnconfigure(1, weight=1)
        ttk.Label(
            action_bar,
            text="Simulation Results",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(action_bar, text="Run Simulation", command=self.start_auto_feed).grid(
            row=0,
            column=1,
            padx=(8, 0),
        )
        ttk.Button(action_bar, text="Reset", command=self.reset_trial).grid(
            row=0,
            column=2,
            padx=(8, 0),
        )
        ttk.Button(action_bar, text="Export CSV + Chart", command=self.export_csv).grid(
            row=0,
            column=3,
            padx=(8, 0),
        )
        ttk.Button(action_bar, text="Run All Shapes + Export", command=self.run_all_shapes_and_export).grid(
            row=0,
            column=4,
            padx=(8, 0),
        )
        ttk.Button(action_bar, text="Consolidate Graph", command=self.consolidate_all_graphs).grid(
            row=0,
            column=5,
            padx=(8, 0),
        )
        ttk.Label(action_bar, text="Save folder").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(action_bar, textvariable=self.output_dir_var).grid(
            row=1,
            column=1,
            columnspan=3,
            sticky="ew",
            padx=(8, 0),
            pady=(8, 0),
        )
        ttk.Button(action_bar, text="Browse", command=self.browse_output_dir).grid(
            row=1,
            column=4,
            padx=(8, 0),
            pady=(8, 0),
        )

        summary = ttk.LabelFrame(results, text="Latest Global Shape Confidence", padding=8)
        summary.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        summary.columnconfigure(0, weight=1)
        self.summary_tree = ttk.Treeview(
            summary,
            columns=("Algorithm", "Prediction", "Confidence", "Stop"),
            show="headings",
            height=6,
        )
        for column, width, anchor in (
            ("Algorithm", 170, tk.W),
            ("Prediction", 150, tk.W),
            ("Confidence", 100, tk.E),
            ("Stop", 120, tk.W),
        ):
            self.summary_tree.heading(column, text=column)
            self.summary_tree.column(column, width=width, anchor=anchor)
        self.summary_tree.pack(fill=tk.X)

        chart_frame = ttk.LabelFrame(results, text="Rule-Based Confidence Chart", padding=8)
        chart_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        chart_frame.columnconfigure(0, weight=1)
        self.chart_figure = Figure(figsize=(7.2, 2.4), dpi=100)
        self.chart_axis = self.chart_figure.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.chart_figure, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.X, expand=True)
        self.refresh_chart()

        history_frame = ttk.LabelFrame(results, text="Touch History", padding=8)
        history_frame.grid(row=3, column=0, sticky="nsew")
        history_frame.rowconfigure(0, weight=1)
        history_frame.columnconfigure(0, weight=1)
        self.history_tree = ttk.Treeview(
            history_frame,
            columns=("Touch", "Feature"),
            show="headings",
            height=16,
        )
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.configure_history_columns()

    def algorithm_column_id(self, spec) -> str:
        return f"algo_{spec.key}"

    def configure_history_columns(self) -> None:
        if not hasattr(self, "history_tree"):
            return
        columns = ["Trial", "Touch", "Feature"]
        for spec, _algorithm in self.selected_algorithms():
            columns.append(self.algorithm_column_id(spec))
        self.history_tree.configure(columns=columns)
        self.history_tree.heading("Trial", text="Trial")
        self.history_tree.column("Trial", width=60, anchor=tk.CENTER, stretch=False)
        self.history_tree.heading("Touch", text="Touch #")
        self.history_tree.column("Touch", width=70, anchor=tk.CENTER, stretch=False)
        self.history_tree.heading("Feature", text="Feature")
        self.history_tree.column("Feature", width=160, anchor=tk.W, stretch=True)
        for spec, _algorithm in self.selected_algorithms():
            column = self.algorithm_column_id(spec)
            self.history_tree.heading(column, text=spec.display_name)
            self.history_tree.column(column, width=190, anchor=tk.W, stretch=True)

    def confidence_chart_series(self) -> dict[str, list[tuple[int, float]]]:
        grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in self.history:
            touch = int(row["touch"])
            for algorithm_name, result in row["algorithm_results"].items():
                grouped[algorithm_name][touch].append(
                    float(result["shape_confidence_percent"])
                )
        return {
            algorithm_name: [
                (touch, sum(values) / len(values))
                for touch, values in sorted(touch_groups.items())
            ]
            for algorithm_name, touch_groups in grouped.items()
        }

    def draw_confidence_chart(
        self,
        axis,
        series: dict[str, list[tuple[int, float]]],
        title: str,
    ) -> None:
        axis.clear()
        if not series:
            axis.set_title(title)
            axis.text(
                0.5,
                0.5,
                "Run a simulation to view algorithm confidence.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            return

        max_touch = max(
            touch
            for points in series.values()
            for touch, _confidence in points
        )
        for index, (algorithm_name, points) in enumerate(sorted(series.items())):
            touches = [touch for touch, _confidence in points]
            confidence = [value for _touch, value in points]
            axis.plot(
                touches,
                confidence,
                color=ALGORITHM_COLORS.get(algorithm_name, "#374151"),
                linestyle=ALGORITHM_STYLES[index % len(ALGORITHM_STYLES)],
                marker=ALGORITHM_MARKERS[index % len(ALGORITHM_MARKERS)],
                linewidth=2.0,
                markersize=3.2,
                label=algorithm_name,
            )
        axis.axhline(50, color="#9ca3af", linestyle=":", linewidth=1.0, alpha=0.9)
        axis.axhline(90, color="#4b5563", linestyle="--", linewidth=1.0, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("Touch")
        axis.set_ylabel("Confidence (%)")
        axis.set_xlim(1, max(1, max_touch))
        axis.set_ylim(0, 105)
        axis.set_xticks(list(range(3, max_touch + 1, 3)) or [1])
        axis.set_yticks([0, 20, 40, 50, 60, 80, 90, 100])
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        axis.legend(fontsize=7, ncol=2)

    def refresh_chart(self) -> None:
        if not hasattr(self, "chart_canvas"):
            return
        title = (
            f"{display_label(self.shape_var.get())} - "
            f"{self.order_mode_var.get()} mean confidence"
        )
        self.draw_confidence_chart(self.chart_axis, self.confidence_chart_series(), title)
        self.chart_figure.tight_layout()
        self.chart_canvas.draw_idle()

    def save_chart_png(self, file_path: Path, shape: str) -> None:
        figure = Figure(figsize=(7.2, 4.2), dpi=300)
        axis = figure.add_subplot(111)
        title = f"{display_label(shape)} - {self.order_mode_var.get()} mean confidence"
        self.draw_confidence_chart(axis, self.confidence_chart_series(), title)
        figure.tight_layout()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(file_path)

    def parse_simulation_filename(self, file_path: Path) -> tuple[str, str]:
        match = SIMULATION_FILENAME_RE.match(file_path.name)
        if not match:
            raise ValueError(f"Cannot infer shape and order from {file_path.name}")
        return match.group("shape"), match.group("order")

    def parse_confidence_percent(self, value: str) -> float:
        match = CONFIDENCE_RE.search(str(value))
        if not match:
            raise ValueError(f"Cannot parse confidence value: {value!r}")
        return float(match.group("confidence"))

    def read_simulation_csvs(self, csv_paths: list[Path]) -> list[dict[str, object]]:
        rows = []
        for csv_path in csv_paths:
            shape, order = self.parse_simulation_filename(csv_path)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or ())
                required = {"trial", "touch", "feature"}
                missing = required - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")
                confidence_columns = sorted(
                    column for column in fieldnames if column.endswith("_global_shape_confidence")
                )
                if "global_shape_confidence" in fieldnames:
                    confidence_columns.insert(0, "global_shape_confidence")
                if not confidence_columns:
                    raise ValueError(f"{csv_path.name} has no algorithm confidence columns")
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
                                or algorithm_name_from_confidence_column(column),
                                "touch": int(row["touch"]),
                                "confidence": self.parse_confidence_percent(confidence),
                            }
                        )
        return rows

    def summarize_simulation_rows(
        self,
        rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
        for row in rows:
            grouped[
                (
                    str(row["shape"]),
                    str(row["order"]),
                    str(row["algorithm"]),
                    int(row["touch"]),
                )
            ].append(float(row["confidence"]))
        return [
            {
                "shape": shape,
                "order": order,
                "algorithm": algorithm,
                "touch": touch,
                "confidence": sum(values) / len(values),
            }
            for (shape, order, algorithm, touch), values in sorted(grouped.items())
        ]

    def draw_consolidated_chart(
        self,
        axis,
        summary_rows: list[dict[str, object]],
        title: str,
    ) -> None:
        axis.clear()
        series: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
        for row in summary_rows:
            series[(str(row["shape"]), str(row["order"]), str(row["algorithm"]))].append(row)

        if not series:
            axis.set_title(title)
            axis.text(
                0.5,
                0.5,
                "No exported simulation CSVs were found.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            return

        max_touch = max(
            int(row["touch"])
            for rows in series.values()
            for row in rows
        )
        algorithms = sorted({algorithm for _shape, _order, algorithm in series})
        orders = sorted({order for _shape, order, _algorithm in series})
        for (shape, order, algorithm), rows in sorted(series.items()):
            ordered = sorted(rows, key=lambda row: int(row["touch"]))
            touches = [int(row["touch"]) for row in ordered]
            confidence = [float(row["confidence"]) for row in ordered]
            style_index = algorithms.index(algorithm) if algorithm in algorithms else 0
            label = f"{display_label(shape)} - {algorithm}"
            if len(orders) > 1:
                label += f" ({order.replace('_', ' ')})"
            axis.plot(
                touches,
                confidence,
                color=SHAPE_COLORS.get(shape, "#374151"),
                linestyle=ALGORITHM_STYLES[style_index % len(ALGORITHM_STYLES)],
                marker=ALGORITHM_MARKERS[style_index % len(ALGORITHM_MARKERS)],
                linewidth=2.0,
                markersize=2.6,
                label=label,
            )

        axis.axhline(50, color="#9ca3af", linestyle=":", linewidth=1.0, alpha=0.9)
        axis.axhline(90, color="#4b5563", linestyle="--", linewidth=1.0, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("Touch")
        axis.set_ylabel("Confidence (%)")
        axis.set_xlim(1, max(1, max_touch))
        axis.set_ylim(0, 105)
        axis.set_xticks(list(range(3, max_touch + 1, 3)) or [1])
        axis.set_yticks([0, 20, 40, 50, 60, 80, 90, 100])
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        axis.legend(fontsize=6.5, ncol=2)

    def save_consolidated_graph(
        self,
        csv_paths: list[Path],
        output_path: Path,
    ) -> Path:
        rows = self.read_simulation_csvs(csv_paths)
        summary_rows = self.summarize_simulation_rows(rows)
        title = "Consolidated Shape Confidence"

        self.draw_consolidated_chart(self.chart_axis, summary_rows, title)
        self.chart_figure.tight_layout()
        self.chart_canvas.draw_idle()

        figure = Figure(figsize=(9.0, 5.2), dpi=300)
        axis = figure.add_subplot(111)
        self.draw_consolidated_chart(axis, summary_rows, title)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
        return output_path

    def on_algorithm_selection_changed(self) -> None:
        self.reset_trial()
        self.configure_history_columns()

    def apply_shape_preset(self) -> None:
        selected = self.presets.get(self.shape_var.get(), set())
        for feature, var in self.feature_vars.items():
            var.set(feature in selected)
        self.reset_trial()

    def select_shape_features(self, shape: str) -> list[str]:
        selected = self.presets.get(shape, set())
        self.shape_var.set(shape)
        for feature, var in self.feature_vars.items():
            var.set(feature in selected)
        return [feature for feature in self.features if feature in selected]

    def clear_features(self) -> None:
        for var in self.feature_vars.values():
            var.set(False)
        self.stop_auto_feed()

    def selected_features(self) -> list[str]:
        return [feature for feature in self.features if self.feature_vars[feature].get()]

    def selected_algorithms(self):
        return [
            (spec, algorithm)
            for key, (spec, algorithm) in self.algorithms.items()
            if self.algorithm_vars.get(key) is not None
            and self.algorithm_vars[key].get()
        ]

    def touch_count(self) -> int:
        try:
            return max(1, int(self.touch_count_var.get()))
        except (tk.TclError, ValueError):
            self.touch_count_var.set(1)
            return 1

    def trial_count(self) -> int:
        try:
            return max(1, int(self.trial_count_var.get()))
        except (tk.TclError, ValueError):
            self.trial_count_var.set(1)
            return 1

    def choose_next_feature(self, features: list[str]) -> str:
        mode = self.order_mode_var.get()
        if mode == "Random":
            return random.choice(features)
        if mode == "Random unique first":
            used = {
                str(row["feature"])
                for row in self.history
                if int(row["trial"]) == self.current_trial
            }
            remaining = [feature for feature in features if feature not in used]
            return random.choice(remaining or features)
        return features[self.touch_counter % len(features)]

    def build_touch_sequence(self, features: list[str], count: int) -> list[str]:
        mode = self.order_mode_var.get()
        if mode == "Random":
            return [random.choice(features) for _ in range(count)]
        if mode == "Random unique first":
            sequence = []
            while len(sequence) < count:
                chunk = list(features)
                random.shuffle(chunk)
                sequence.extend(chunk)
            return sequence[:count]
        return [features[index % len(features)] for index in range(count)]

    def build_trial_queue(self, features: list[str]) -> list[dict[str, object]]:
        queue = []
        for trial in range(1, self.trial_count() + 1):
            for feature in self.build_touch_sequence(features, self.touch_count()):
                queue.append({"trial": trial, "feature": feature})
        return queue

    def reset_algorithms(self) -> None:
        for _, algorithm in self.algorithms.values():
            if hasattr(algorithm, "reset"):
                algorithm.reset()

    def reset_trial(self) -> None:
        self.stop_auto_feed()
        self.current_trial = 1
        self.touch_counter = 0
        self.history = []
        self.auto_queue = []
        self.reset_algorithms()
        for tree in (getattr(self, "history_tree", None), getattr(self, "summary_tree", None)):
            if tree is None:
                continue
            for item in tree.get_children():
                tree.delete(item)
        self.refresh_chart()
        self.status_var.set("Ready")

    def feed_next_touch(self) -> None:
        features = self.selected_features()
        if not features:
            messagebox.showinfo("Simulation", "Select at least one local feature.")
            return
        if not self.history:
            self.current_trial = 1
        next_feature = self.choose_next_feature(features)
        self.feed_touch(next_feature, trial=self.current_trial)

    def start_auto_feed(self) -> None:
        features = self.selected_features()
        if not features:
            messagebox.showinfo("Simulation", "Select at least one local feature.")
            return
        if not self.selected_algorithms():
            messagebox.showinfo("Simulation", "Select at least one algorithm.")
            return
        self.reset_trial()
        self.auto_queue = self.build_trial_queue(features)
        self.auto_running = True
        self.status_var.set(
            f"Running {self.trial_count()} trial(s), {self.touch_count()} touch(es) each..."
        )
        self.run_auto_step()

    def run_auto_step(self) -> None:
        if not self.auto_running:
            return
        if not self.auto_queue:
            self.auto_running = False
            self.status_var.set("Auto feed complete")
            return
        item = self.auto_queue.pop(0)
        trial = int(item["trial"])
        if trial != self.current_trial:
            self.current_trial = trial
            self.touch_counter = 0
            self.reset_algorithms()
        feature = str(item["feature"])
        self.feed_touch(feature, trial=trial)
        delay = max(100, int(self.interval_var.get()))
        self.root.after(delay, self.run_auto_step)

    def stop_auto_feed(self) -> None:
        self.auto_running = False

    def browse_output_dir(self) -> None:
        current = self.output_dir_var.get().strip()
        initialdir = current if current and os.path.isdir(current) else str(PROJECT_ROOT)
        selected = filedialog.askdirectory(
            initialdir=initialdir,
            title="Select CSV Save Folder",
        )
        if selected:
            self.output_dir_var.set(selected)

    def output_dir(self) -> Path:
        selected = self.output_dir_var.get().strip()
        return Path(selected).expanduser() if selected else PROJECT_ROOT / "output" / "shape_simulations"

    def feed_touch(self, feature: str, trial: int | None = None) -> None:
        algorithms = self.selected_algorithms()
        if not algorithms:
            messagebox.showinfo("Simulation", "Select at least one algorithm.")
            return

        if trial is None:
            trial = self.current_trial
        self.current_trial = int(trial)
        self.touch_counter += 1
        feature_probabilities = {
            name: 1.0 if name == feature else 0.0
            for name in self.features
        }
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        latest_rows = []
        touch_results = {}

        for spec, algorithm in algorithms:
            try:
                result = spec.update(
                    algorithm,
                    feature_probabilities=feature_probabilities,
                    top_feature=feature,
                )
                prediction, confidence, probabilities, should_stop, is_uncertain, stopping_reason = (
                    self.summarize_result(result)
                )
            except Exception as exc:
                prediction = "error"
                confidence = 0.0
                probabilities = {}
                should_stop = False
                is_uncertain = False
                stopping_reason = str(exc)

            display_confidence = f"{display_label(prediction)} ({confidence:.1f}%)"
            row = {
                "timestamp": timestamp,
                "algorithm": spec.display_name,
                "predicted_shape": prediction,
                "shape_confidence_percent": confidence,
                "should_stop": should_stop,
                "is_uncertain": is_uncertain,
                "stopping_reason": stopping_reason,
                "shape_probabilities": probabilities,
            }
            touch_results[spec.display_name] = row
            latest_rows.append(row)

        values = [self.current_trial, self.touch_counter, display_label(feature)]
        for spec, _algorithm in algorithms:
            result = touch_results.get(spec.display_name, {})
            values.append(
                f"{display_label(str(result.get('predicted_shape', 'unknown')))} "
                f"({float(result.get('shape_confidence_percent', 0.0)):.1f}%)"
            )
        self.history_tree.insert("", tk.END, values=values)
        self.history.append(
            {
                "trial": self.current_trial,
                "touch": self.touch_counter,
                "timestamp": timestamp,
                "feature": feature,
                "algorithm_results": touch_results,
            }
        )

        self.history_tree.yview_moveto(1)
        self.update_summary(latest_rows)
        self.refresh_chart()
        self.status_var.set(
            f"Trial {self.current_trial}, touch {self.touch_counter}: {display_label(feature)}"
        )

    def summarize_result(self, result) -> tuple[str, float, dict[str, float], bool, bool, str]:
        if isinstance(result, dict):
            probabilities = {str(name): float(prob) for name, prob in result.items()}
            prediction = max(probabilities, key=probabilities.get)
            return prediction, probabilities[prediction] * 100.0, probabilities, False, False, ""

        belief = getattr(result, "belief", None) or {}
        probabilities = {str(name): float(prob) for name, prob in belief.items()}
        prediction = getattr(result, "prediction", None)
        if not prediction and probabilities:
            prediction = max(probabilities, key=probabilities.get)
        prediction = str(prediction or "unknown")
        confidence = probabilities.get(prediction, 0.0) * 100.0
        return (
            prediction,
            confidence,
            probabilities,
            bool(getattr(result, "should_stop", False)),
            bool(getattr(result, "is_uncertain", False)),
            str(getattr(result, "stopping_reason", "") or ""),
        )

    def update_summary(self, latest_rows: list[dict[str, object]]) -> None:
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        for row in latest_rows:
            stop_state = "uncertain" if row["is_uncertain"] else "accepted" if row["should_stop"] else ""
            if not stop_state:
                stop_state = str(row["stopping_reason"]) if row["stopping_reason"] else "-"
            self.summary_tree.insert(
                "",
                tk.END,
                values=(
                    row["algorithm"],
                    display_label(str(row["predicted_shape"])),
                    f"{float(row['shape_confidence_percent']):.1f}%",
                    stop_state,
                ),
            )

    def export_csv(self) -> None:
        if not self.history:
            messagebox.showinfo("Export CSV", "No simulated touches to export.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shape_name = slug(self.shape_var.get())
        order_name = slug(self.order_mode_var.get())
        default_name = (
            f"shape_sim_{shape_name}_{order_name}_"
            f"{self.trial_count()}trials_{self.touch_count()}touches_{timestamp}.csv"
        )
        file_path = filedialog.asksaveasfilename(
            title="Export Shape Simulation CSV and Chart",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
        )
        if not file_path:
            return

        try:
            csv_path = Path(file_path)
            chart_path = csv_path.with_suffix(".png")
            self.write_history_csv(csv_path, self.shape_var.get(), self.selected_features())
            self.save_chart_png(chart_path, self.shape_var.get())
        except Exception as exc:
            messagebox.showerror("Export", f"Failed to export files:\n\n{exc}")
            return

        self.status_var.set(f"Exported {os.path.basename(file_path)} and chart")
        messagebox.showinfo(
            "Export",
            f"Saved CSV:\n{csv_path}\n\nSaved chart:\n{chart_path}",
        )

    def write_history_csv(
        self,
        file_path: Path,
        shape: str,
        selected_features: list[str],
    ) -> None:
        algorithm_names = []
        for row in self.history:
            for algorithm_name in row["algorithm_results"]:
                if algorithm_name not in algorithm_names:
                    algorithm_names.append(algorithm_name)
        single_algorithm = len(algorithm_names) == 1
        fieldnames = [
            "trial",
            "touch",
            "feature",
        ]
        if single_algorithm:
            fieldnames.append("algorithm")
            fieldnames.append("global_shape_confidence")
        else:
            fieldnames.extend(confidence_column_name(name) for name in algorithm_names)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                output = {
                    "trial": row["trial"],
                    "touch": row["touch"],
                    "feature": row["feature"],
                }
                if single_algorithm:
                    output["algorithm"] = algorithm_names[0] if algorithm_names else ""
                for algorithm_name in algorithm_names:
                    result = row["algorithm_results"].get(algorithm_name)
                    column = confidence_column_name(
                        algorithm_name,
                        single_algorithm=single_algorithm,
                    )
                    output[column] = ""
                    if result is not None:
                        output[column] = (
                            f"{result['predicted_shape']} "
                            f"({float(result['shape_confidence_percent']):.1f}%)"
                        )
                writer.writerow(output)

    def run_simulation_immediately(self, shape: str, features: list[str]) -> None:
        self.reset_trial()
        self.shape_var.set(shape)
        for trial in range(1, self.trial_count() + 1):
            self.current_trial = trial
            self.touch_counter = 0
            self.reset_algorithms()
            for feature in self.build_touch_sequence(features, self.touch_count()):
                self.feed_touch(feature, trial=trial)

    def optimize_modified_nb_plan(self) -> None:
        shape = self.shape_var.get()
        features = self.selected_features()
        if not shape or not features:
            messagebox.showinfo(
                "Optimize Modified NB",
                "Select a shape and at least one local feature first.",
            )
            return

        try:
            from algorithms.modified_naive_bayes.factory import create_algorithm
            from tools.optimize_modified_nb_touch_plan import optimize_shape

            modified_nb = create_algorithm()
            plan = optimize_shape(
                modified_nb.config,
                shape,
                candidate_features=features,
                threshold=0.90,
                max_touches=max(1, self.touch_count()),
                beam_width=250,
                prefer_unique_until_covered=True,
            )
        except Exception as exc:
            messagebox.showerror(
                "Optimize Modified NB",
                f"Failed to optimize touch plan:\n\n{exc}",
            )
            return

        variable = self.algorithm_vars.get("modified_naive_bayes")
        if variable is not None:
            variable.set(True)
            self.configure_history_columns()

        self.reset_trial()
        self.shape_var.set(shape)
        for feature in plan.sequence:
            self.feed_touch(feature, trial=1)

        threshold_text = (
            f"reached 90% confidence at touch {plan.threshold_touch}"
            if plan.threshold_touch is not None
            else f"did not reach 90% within {self.touch_count()} touches"
        )
        self.status_var.set(
            f"Optimized Modified NB plan for {display_label(shape)}: {threshold_text}"
        )
        messagebox.showinfo(
            "Optimize Modified NB",
            f"{display_label(shape)} {threshold_text}.\n\n"
            f"Sequence:\n{' -> '.join(display_label(feature) for feature in plan.sequence)}",
        )

    def run_all_shapes_and_export(self) -> None:
        if not self.selected_algorithms():
            messagebox.showinfo("Run All Shapes", "Select at least one algorithm.")
            return

        output_dir = self.output_dir()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exported = []

        for shape in self.shapes:
            features = self.select_shape_features(shape)
            if not features:
                continue
            self.run_simulation_immediately(shape, features)
            output_path = output_dir / (
                f"shape_sim_{slug(shape)}_{slug(self.order_mode_var.get())}_"
                f"{self.trial_count()}trials_{self.touch_count()}touches_{timestamp}.csv"
            )
            self.write_history_csv(output_path, shape, features)
            self.save_chart_png(output_path.with_suffix(".png"), shape)
            exported.append(output_path)

        if not exported:
            messagebox.showinfo("Run All Shapes", "No shape presets were available to export.")
            return

        consolidated_path = output_dir / f"consolidated_shape_confidence_{timestamp}.png"
        self.save_consolidated_graph(exported, consolidated_path)

        self.status_var.set(f"Exported {len(exported)} shape CSV/chart pairs and consolidated graph")
        messagebox.showinfo(
            "Run All Shapes",
            "Saved CSV and PNG files to:\n"
            + str(output_dir)
            + "\n\nConsolidated graph:\n"
            + str(consolidated_path),
        )

    def consolidate_all_graphs(self) -> None:
        output_dir = self.output_dir()
        csv_paths = sorted(
            path
            for path in output_dir.glob("shape_sim_*.csv")
            if path.is_file()
        )
        if not csv_paths:
            messagebox.showinfo(
                "Consolidate Graph",
                f"No shape_sim_*.csv files found in:\n{output_dir}",
            )
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"consolidated_shape_confidence_{timestamp}.png"
        try:
            self.save_consolidated_graph(csv_paths, output_path)
        except Exception as exc:
            messagebox.showerror(
                "Consolidate Graph",
                f"Failed to consolidate graph:\n\n{exc}",
            )
            return

        self.status_var.set(f"Consolidated {len(csv_paths)} CSV files")
        messagebox.showinfo(
            "Consolidate Graph",
            f"Saved consolidated graph:\n{output_path}",
        )


def main() -> int:
    root = tk.Tk()
    ShapeSimulationApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
