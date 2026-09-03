"""Unified optimization, evaluation, and reporting for shape classifiers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.bag_of_features.algorithm import BagOfFeaturesClassifier
from algorithms.bag_of_features.config import (
    BagOfFeaturesConfig,
    load_config as load_bof_config,
)
from algorithms.bag_of_features.optimization import optimize as optimize_bof
from algorithms.bayesian.algorithm import BayesianShapePredictor
from algorithms.bayesian.optimization import optimize_bayesian
from algorithms.dirichlet_multinomial.algorithm import DirichletMultinomialClassifier
from algorithms.dirichlet_multinomial.config import (
    DirichletMultinomialConfig,
    load_config as load_dm_config,
)
from algorithms.dirichlet_multinomial.optimization import optimize as optimize_dm
from algorithms.hard_label_optimization import (
    evaluate_prefixes,
    install_algorithm_default,
    load_trials,
    normalize_trials,
    ordered_runs,
    timestamp,
    write_json,
)
from algorithms.modified_naive_bayes.algorithm import ModifiedNaiveBayesClassifier
from algorithms.modified_naive_bayes.config import (
    ModifiedNaiveBayesConfig,
    load_config as load_modified_nb_config,
)
from algorithms.modified_naive_bayes.optimization import optimize as optimize_modified_nb
from algorithms.naive_bayes.algorithm import NaiveBayesClassifier
from algorithms.naive_bayes.config import (
    NaiveBayesConfig,
    load_config as load_nb_config,
)
from algorithms.naive_bayes.optimization import optimize as optimize_nb
from algorithms.rule_based.algorithm import RuleBasedClassifier
from algorithms.rule_based.config import (
    RuleBasedConfig,
    load_config as load_rule_based_config,
)
from algorithms.rule_based.optimization import optimize as optimize_rule_based
from algorithms.rfs.config import RFSConfig, load_rfs_config
from algorithms.rfs.optimization import evaluate_frozen_config, optimize_rfs_config
from algorithms.single_touch_baseline.algorithm import SingleTouchBaselineClassifier
from algorithms.single_touch_baseline.config import (
    SingleTouchBaselineConfig,
    load_config as load_single_touch_config,
)
from algorithms.single_touch_baseline.optimization import optimize as optimize_single_touch


ALGORITHMS = (
    ("bayesian", "Bayesian"),
    ("naive_bayes", "Naive Bayes"),
    ("modified_naive_bayes", "Modified Naive Bayes"),
    ("single_touch_baseline", "Single Touch Baseline"),
    ("rule_based", "Rule-Based"),
    ("bag_of_features", "Bag of Features"),
    ("dirichlet_multinomial", "Dirichlet-Multinomial"),
    ("rfs", "Set Evidence (DSEA)"),
)
DEFAULT_TRAIN = PROJECT_ROOT / "trials" / "rfs_train.json"
DEFAULT_VALIDATION = PROJECT_ROOT / "trials" / "rfs_validation.json"
DEFAULT_TEST = PROJECT_ROOT / "trials" / "rfs_test.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "optimize"


@dataclass(frozen=True)
class RunRequest:
    model: Path
    train: Path
    validation: Path
    test: Path | None
    algorithms: tuple[str, ...]
    output_root: Path = DEFAULT_OUTPUT
    permutations: int = 5
    test_permutations: int = 20
    seed: int = 42
    install_sidecars: bool = True
    apply_algorithm_defaults: bool = True


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return normalized or "model"


def allocate_run_directory(output_root: Path, model: Path) -> Path:
    model_root = Path(output_root).expanduser().resolve() / _safe_name(model.stem)
    model_root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1000000):
        candidate = model_root / f"run_{number:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a run directory under {model_root}")


def _copy_snapshot(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _flatten(value, prefix=""):
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_flatten(item, f"{prefix}[{index}]"))
    elif value is None or isinstance(value, (str, int, float, bool)):
        rows.append((prefix, value))
    return rows


def _write_metrics_csv(path: Path, report: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(_flatten(report))
    return path


def _generic_test_metrics(trials, config, classifier_type, permutations, seed):
    parsed = normalize_trials(
        trials,
        features=config.features,
        shapes=config.shapes,
        aliases=config.aliases,
        name="Test",
    )
    runs = ordered_runs(parsed, permutations=permutations, seed=seed)
    return evaluate_prefixes(
        runs,
        shapes=config.shapes,
        create_classifier=lambda: classifier_type(config),
    )


def _bayesian_classifier(config):
    predictor = BayesianShapePredictor.__new__(BayesianShapePredictor)
    predictor.config_path = ""
    predictor.SHAPE_LIKELIHOODS = config
    predictor.reset()
    return predictor


def _bayesian_test_metrics(trials, config, features, shapes, aliases, permutations, seed):
    parsed = normalize_trials(
        trials,
        features=features,
        shapes=shapes,
        aliases=aliases,
        name="Test",
    )
    runs = ordered_runs(parsed, permutations=permutations, seed=seed)
    return evaluate_prefixes(
        runs,
        shapes=shapes,
        create_classifier=lambda: _bayesian_classifier(config),
    )


def _headline(report: dict) -> dict:
    validation = report.get("validation_metrics")
    if validation and "prefix" in validation:
        validation = validation["prefix"]
    elif "validation_after_temperature_calibration" in report:
        validation = report["validation_after_temperature_calibration"]
    else:
        validation = {}
    test = report.get("test_metrics") or report.get("test", {}).get("metrics", {})
    if "prefix" in test:
        test_summary = test["prefix"]
    else:
        test_summary = test
    test_accuracy = None
    if isinstance(test_summary, dict):
        test_accuracy = test_summary.get("macro_accuracy")
        if test_accuracy is None:
            test_accuracy = test_summary.get("overall_accuracy")
    return {
        "validation_accuracy": validation.get("macro_accuracy"),
        "validation_nll": validation.get("macro_nll"),
        "test_accuracy": test_accuracy,
    }


def _curve(report: dict, split: str):
    if split == "validation":
        metrics = report.get("validation_metrics", {})
    else:
        metrics = report.get("test_metrics") or report.get("test", {}).get("metrics", {})
    return metrics.get("accuracy_by_touch_count", {}) if isinstance(metrics, dict) else {}


def _generate_charts(run_dir: Path, reports: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = run_dir / "charts"
    charts.mkdir(exist_ok=True)
    outputs = []

    names = list(reports)
    summaries = [_headline(reports[name]) for name in names]
    val_accuracy = [summary["validation_accuracy"] or 0.0 for summary in summaries]
    test_accuracy = [summary["test_accuracy"] or 0.0 for summary in summaries]
    x = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.36
    ax.bar([item - width / 2 for item in x], val_accuracy, width, label="Validation")
    if any(summary["test_accuracy"] is not None for summary in summaries):
        ax.bar([item + width / 2 for item in x], test_accuracy, width, label="Test")
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Algorithm Accuracy Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = charts / f"accuracy_comparison.{suffix}"
        fig.savefig(path, dpi=180)
        outputs.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    has_curve = False
    for name, report in reports.items():
        curve = _curve(report, "test") or _curve(report, "validation")
        if not curve:
            continue
        points = sorted((int(key), value["accuracy"]) for key, value in curve.items())
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=name)
        has_curve = True
    if has_curve:
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Available touches")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Touch Count")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = charts / f"accuracy_by_touch_count.{suffix}"
            fig.savefig(path, dpi=180)
            outputs.append(path)
    plt.close(fig)

    for name, report in reports.items():
        metrics = report.get("test_metrics") or report.get("test", {}).get("metrics", {})
        confusion = metrics.get("confusion_matrix", {}) if isinstance(metrics, dict) else {}
        if not confusion and isinstance(metrics, dict) and metrics.get("runs"):
            labels = sorted(
                {run["true_shape"] for run in metrics["runs"]}
                | {run["prediction"] for run in metrics["runs"]}
            )
            confusion = {true: {predicted: 0 for predicted in labels} for true in labels}
            for run in metrics["runs"]:
                confusion[run["true_shape"]][run["prediction"]] += 1
        if not confusion:
            continue
        labels = list(confusion)
        matrix = [[confusion[true].get(predicted, 0) for predicted in labels] for true in labels]
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted shape")
        ax.set_ylabel("True shape")
        ax.set_title(f"{name} Confusion Matrix")
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                ax.text(column, row, str(value), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        stem = _safe_name(name).lower()
        for suffix in ("png", "svg"):
            path = charts / f"confusion_matrix_{stem}.{suffix}"
            fig.savefig(path, dpi=180)
            outputs.append(path)
        plt.close(fig)
    return outputs


def run_optimization(request: RunRequest, progress=lambda *_: None) -> dict:
    model = request.model.expanduser().resolve()
    if not model.is_file() or model.suffix.lower() != ".pth":
        raise FileNotFoundError(f"Select an existing .pth model: {model}")
    if not request.algorithms:
        raise ValueError("Select at least one algorithm")
    if request.permutations < 1 or request.test_permutations < 1:
        raise ValueError("Permutation counts must be positive")

    train_path = request.train.expanduser().resolve()
    validation_path = request.validation.expanduser().resolve()
    test_path = request.test.expanduser().resolve() if request.test else None
    train = load_trials(train_path)
    validation = load_trials(validation_path)
    test = load_trials(test_path) if test_path else None
    run_dir = allocate_run_directory(request.output_root, model)
    inputs = run_dir / "inputs"
    _copy_snapshot(train_path, inputs / "train.json")
    _copy_snapshot(validation_path, inputs / "validation.json")
    if test_path:
        _copy_snapshot(test_path, inputs / "test.json")

    rfs_base = load_rfs_config(PROJECT_ROOT / "algorithms" / "rfs" / "config.json")
    reports = {}
    artifacts = {}
    for index, key in enumerate(request.algorithms, start=1):
        display = dict(ALGORITHMS)[key]
        progress(key, "running", f"Optimizing {display} ({index}/{len(request.algorithms)})")
        algorithm_dir = run_dir / key
        algorithm_dir.mkdir()
        run_name = f"{model.stem}_{key}_{run_dir.name}"

        if key == "bayesian":
            optimized, report = optimize_bayesian(
                train,
                validation,
                features=rfs_base.features,
                shapes=rfs_base.shapes,
                aliases=rfs_base.aliases,
                permutations=request.permutations,
                seed=request.seed,
            )
            if test:
                report["test_metrics"] = _bayesian_test_metrics(
                    test,
                    optimized,
                    rfs_base.features,
                    rfs_base.shapes,
                    rfs_base.aliases,
                    request.test_permutations,
                    request.seed,
                )
            suffix = ".bayesian.json"
        elif key == "naive_bayes":
            base = load_nb_config(PROJECT_ROOT / "algorithms" / key / "config.json")
            optimized, report = optimize_nb(
                train,
                validation,
                base,
                smoothing_values=(0.01, 0.05, 0.1, 0.5, 1.0),
                permutations=request.permutations,
                seed=request.seed,
                name=run_name,
            )
            config = NaiveBayesConfig.from_mapping(optimized)
            if test:
                report["test_metrics"] = _generic_test_metrics(
                    test,
                    config,
                    NaiveBayesClassifier,
                    request.test_permutations,
                    request.seed,
                )
            suffix = ".nb.json"
        elif key == "modified_naive_bayes":
            base = load_modified_nb_config(PROJECT_ROOT / "algorithms" / key / "config.json")
            optimized, report = optimize_modified_nb(
                train,
                validation,
                base,
                smoothing_values=(0.01, 0.05, 0.1, 0.5, 1.0),
                repeat_decay_values=(0.0, 0.5, 1.0),
                coverage_weight_values=(0.0, 0.5, 1.0),
                unexpected_penalty_values=(0.0, 0.5, 1.0),
                feature_weight_strength_values=(0.0, 0.5, 1.0),
                coverage_rate_values=(0.5, 1.0, 2.0),
                permutations=request.permutations,
                seed=request.seed,
                name=run_name,
            )
            config = ModifiedNaiveBayesConfig.from_mapping(optimized)
            if test:
                report["test_metrics"] = _generic_test_metrics(
                    test,
                    config,
                    ModifiedNaiveBayesClassifier,
                    request.test_permutations,
                    request.seed,
                )
            suffix = ".mnb.json"
        elif key == "single_touch_baseline":
            base = load_single_touch_config(PROJECT_ROOT / "algorithms" / key / "config.json")
            optimized, report = optimize_single_touch(
                train,
                validation,
                base,
                smoothing_values=(0.01, 0.05, 0.1, 0.5, 1.0),
                permutations=request.permutations,
                seed=request.seed,
                name=run_name,
            )
            config = SingleTouchBaselineConfig.from_mapping(optimized)
            if test:
                report["test_metrics"] = _generic_test_metrics(
                    test,
                    config,
                    SingleTouchBaselineClassifier,
                    request.test_permutations,
                    request.seed,
                )
            suffix = ".single_touch.json"
        elif key == "rule_based":
            base = load_rule_based_config(PROJECT_ROOT / "algorithms" / key / "config.json")
            optimized, report = optimize_rule_based(
                train,
                validation,
                base,
                smoothing_values=(0.05, 0.1, 0.5, 1.0),
                repetition_values=(0.0, 0.1, 0.25, 0.5),
                permutations=request.permutations,
                seed=request.seed,
                name=run_name,
            )
            config = RuleBasedConfig.from_mapping(optimized)
            if test:
                report["test_metrics"] = _generic_test_metrics(
                    test,
                    config,
                    RuleBasedClassifier,
                    request.test_permutations,
                    request.seed,
                )
            suffix = ".rules.json"
        elif key == "bag_of_features":
            base = load_bof_config(PROJECT_ROOT / "algorithms" / key / "config.json")
            optimized, report = optimize_bof(
                train,
                validation,
                base,
                smoothing_values=(0.01, 0.05, 0.1, 0.5, 1.0),
                permutations=request.permutations,
                seed=request.seed,
                name=run_name,
            )
            config = BagOfFeaturesConfig.from_mapping(optimized)
            if test:
                report["test_metrics"] = _generic_test_metrics(
                    test, config, BagOfFeaturesClassifier, request.test_permutations, request.seed
                )
            suffix = ".bof.json"
        elif key == "dirichlet_multinomial":
            base = load_dm_config(PROJECT_ROOT / "algorithms" / key / "config.json")
            optimized, report = optimize_dm(
                train,
                validation,
                base,
                permutations=request.permutations,
                seed=request.seed,
                initial_concentration=10.0,
                max_iterations=500,
                name=run_name,
            )
            config = DirichletMultinomialConfig.from_mapping(optimized)
            if test:
                report["test_metrics"] = _generic_test_metrics(
                    test,
                    config,
                    DirichletMultinomialClassifier,
                    request.test_permutations,
                    request.seed,
                )
            suffix = ".dm.json"
        elif key == "rfs":
            optimized, report = optimize_rfs_config(
                train,
                validation,
                rfs_base,
                name=run_name,
                permutations=request.permutations,
                seed=request.seed,
            )
            if test:
                config = RFSConfig.from_mapping(optimized)
                report["test"] = {
                    "warning": "Evaluated after all parameters were frozen.",
                    "metrics": evaluate_frozen_config(
                        test,
                        config,
                        permutations=request.test_permutations,
                        seed=request.seed,
                    ),
                }
            suffix = ".rfs.json"
        else:
            raise ValueError(f"Unknown algorithm: {key}")

        config_path = write_json(algorithm_dir / "optimized_config.json", optimized)
        report["artifacts"] = {
            "config": str(config_path),
            "metrics_csv": str(algorithm_dir / "metrics.csv"),
        }
        report_path = write_json(algorithm_dir / "evaluation_report.json", report)
        csv_path = _write_metrics_csv(algorithm_dir / "metrics.csv", report)
        sidecar = None
        algorithm_default = None
        algorithm_default_backup = None
        if request.install_sidecars:
            sidecar = model.with_suffix(suffix)
            shutil.copy2(config_path, sidecar)
        if request.apply_algorithm_defaults:
            algorithm_default, algorithm_default_backup = install_algorithm_default(
                config_path,
                PROJECT_ROOT / "algorithms" / key / "config.json",
            )
        reports[display] = report
        artifacts[key] = {
            "config": str(config_path),
            "report": str(report_path),
            "metrics_csv": str(csv_path),
            "model_sidecar": str(sidecar) if sidecar else None,
            "algorithm_default": str(algorithm_default) if algorithm_default else None,
            "algorithm_default_backup": (
                str(algorithm_default_backup) if algorithm_default_backup else None
            ),
        }
        progress(key, "completed", f"Completed {display}")

    charts = _generate_charts(run_dir, reports)
    summary_rows = []
    for display, report in reports.items():
        summary = _headline(report)
        summary_rows.append({"algorithm": display, **summary})
    with (run_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("algorithm", "validation_accuracy", "validation_nll", "test_accuracy"),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "schema_version": 1,
        "created_at": timestamp(),
        "model": str(model),
        "run_directory": str(run_dir),
        "inputs": {
            "train": str(train_path),
            "validation": str(validation_path),
            "test": str(test_path) if test_path else None,
            "train_trials": len(train),
            "validation_trials": len(validation),
            "test_trials": len(test) if test else 0,
        },
        "settings": {
            "algorithms": list(request.algorithms),
            "permutations": request.permutations,
            "test_permutations": request.test_permutations,
            "seed": request.seed,
            "install_sidecars": request.install_sidecars,
            "apply_algorithm_defaults": request.apply_algorithm_defaults,
        },
        "summary": summary_rows,
        "algorithm_artifacts": artifacts,
        "charts": [str(path) for path in charts],
    }
    write_json(run_dir / "run_manifest.json", manifest)
    progress("all", "completed", f"Run complete: {run_dir}")
    return manifest


class OptimizationApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("TouchLab Algorithm Optimization")
        self.root.geometry("1060x760")
        self.root.minsize(880, 640)
        self.root.configure(bg="#f5f6f8")
        self.last_run = None
        self.chart_image = None
        self.chart_paths = {}
        self.algorithm_vars = {}
        self.status_items = {}
        self._build_ui()

    def _build_ui(self):
        tk, ttk = self.tk, self.ttk
        header = tk.Frame(self.root, bg="#ffffff", padx=20, pady=14)
        header.pack(fill=tk.X)
        tk.Label(header, text="Algorithm Optimization", bg="#ffffff", fg="#202124", font=("Segoe UI", 17, "bold")).pack(anchor=tk.W)
        tk.Label(header, text="Fit, validate, compare, and export multi-touch shape classifiers", bg="#ffffff", fg="#667085", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(2, 0))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)
        setup = tk.Frame(notebook, bg="#ffffff")
        results = tk.Frame(notebook, bg="#ffffff")
        notebook.add(setup, text="Setup")
        notebook.add(results, text="Results")
        self.results_tab = results

        form = tk.Frame(setup, bg="#ffffff", padx=16, pady=14)
        form.pack(fill=tk.X)
        self.model_var = tk.StringVar(value="")
        self.train_var = tk.StringVar(value=str(DEFAULT_TRAIN))
        self.validation_var = tk.StringVar(value=str(DEFAULT_VALIDATION))
        self.test_var = tk.StringVar(value=str(DEFAULT_TEST) if DEFAULT_TEST.is_file() else "")
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT))
        for row, (label, variable, kind, optional) in enumerate((
            ("Model weights", self.model_var, "file", False),
            ("Training trials", self.train_var, "json", False),
            ("Validation trials", self.validation_var, "json", False),
            ("Test trials", self.test_var, "json", True),
        )):
            tk.Label(form, text=label + (" (optional)" if optional else ""), bg="#ffffff", fg="#344054", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(form, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=5)
            ttk.Button(form, text="Browse", command=lambda v=variable, k=kind: self._browse(v, k)).grid(row=row, column=2, pady=5)
            if optional:
                ttk.Button(form, text="Clear", command=lambda v=variable: v.set("")).grid(row=row, column=3, padx=(5, 0), pady=5)
        tk.Label(form, text="Run output", bg="#ffffff", fg="#344054", font=("Segoe UI", 9, "bold")).grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.output_var, state="readonly").grid(row=4, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(form, text="Browse", command=self._browse_output).grid(row=4, column=2, pady=5)
        form.columnconfigure(1, weight=1)

        options = tk.Frame(setup, bg="#ffffff", padx=16, pady=8)
        options.pack(fill=tk.X)
        algo_box = ttk.LabelFrame(options, text="Algorithms", padding=10)
        algo_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        for key, display in ALGORITHMS:
            variable = tk.BooleanVar(value=True)
            self.algorithm_vars[key] = variable
            ttk.Checkbutton(algo_box, text=display, variable=variable).pack(anchor=tk.W, pady=2)
        settings = ttk.LabelFrame(options, text="Evaluation", padding=10)
        settings.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.permutations_var = tk.IntVar(value=5)
        self.test_permutations_var = tk.IntVar(value=20)
        self.seed_var = tk.IntVar(value=42)
        self.install_var = tk.BooleanVar(value=True)
        self.apply_defaults_var = tk.BooleanVar(value=True)
        for row, (label, variable) in enumerate((
            ("Validation permutations", self.permutations_var),
            ("Test permutations", self.test_permutations_var),
            ("Random seed", self.seed_var),
        )):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Spinbox(settings, from_=1 if row < 2 else 0, to=10000, textvariable=variable, width=10).grid(row=row, column=1, padx=(12, 0), pady=3)
        ttk.Checkbutton(settings, text="Install optimized model sidecars", variable=self.install_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Checkbutton(settings, text="Apply optimized configs to algorithm defaults", variable=self.apply_defaults_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        action = tk.Frame(setup, bg="#ffffff", padx=16, pady=12)
        action.pack(fill=tk.X)
        self.run_button = tk.Button(action, text="Run Optimization", command=self._start, bg="#1769aa", fg="#ffffff", activebackground="#12558a", activeforeground="#ffffff", relief=tk.FLAT, font=("Segoe UI", 10, "bold"), padx=18, pady=9)
        self.run_button.pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(setup, text="Run Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 14))
        self.log = tk.Text(log_frame, height=10, bg="#111827", fg="#d1d5db", insertbackground="#ffffff", font=("Consolas", 9), state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

        result_actions = tk.Frame(results, bg="#ffffff", padx=14, pady=10)
        result_actions.pack(fill=tk.X)
        ttk.Button(result_actions, text="Open Run Folder", command=self._open_run).pack(side=tk.LEFT)
        ttk.Button(result_actions, text="Export Run", command=self._export_run).pack(side=tk.LEFT, padx=6)
        ttk.Button(result_actions, text="View Full Metrics", command=self._view_metrics).pack(side=tk.LEFT)
        self.run_path_var = tk.StringVar(value="No completed run")
        tk.Label(result_actions, textvariable=self.run_path_var, bg="#ffffff", fg="#667085", anchor=tk.W).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.metrics_tree = ttk.Treeview(results, columns=("Algorithm", "ValAcc", "ValNLL", "TestAcc"), show="headings", height=5)
        for column, text, width in (("Algorithm", "Algorithm", 220), ("ValAcc", "Validation Accuracy", 150), ("ValNLL", "Validation NLL", 130), ("TestAcc", "Test Accuracy", 130)):
            self.metrics_tree.heading(column, text=text)
            self.metrics_tree.column(column, width=width, anchor=tk.W if column == "Algorithm" else tk.E)
        self.metrics_tree.pack(fill=tk.X, padx=14, pady=(0, 10))

        chart_bar = tk.Frame(results, bg="#ffffff", padx=14)
        chart_bar.pack(fill=tk.X, pady=(0, 8))
        tk.Label(chart_bar, text="Chart", bg="#ffffff", fg="#344054", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.chart_var = tk.StringVar(value="")
        self.chart_selector = ttk.Combobox(chart_bar, textvariable=self.chart_var, state="readonly", width=54)
        self.chart_selector.pack(side=tk.LEFT, padx=8)
        self.chart_selector.bind("<<ComboboxSelected>>", self._show_selected_chart)

        self.chart_label = tk.Label(results, bg="#f5f6f8", text="Charts appear after optimization", fg="#667085")
        self.chart_label.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

    def _browse(self, variable, kind):
        from tkinter import filedialog
        filetypes = (("PyTorch weights", "*.pth"),) if kind == "file" else (("JSON trials", "*.json"),)
        current = Path(variable.get()).expanduser() if variable.get().strip() else PROJECT_ROOT
        initial = current.parent if current.suffix else current
        path = filedialog.askopenfilename(title="Select file", initialdir=initial, filetypes=filetypes + (("All files", "*.*"),))
        if path:
            variable.set(path)

    def _browse_output(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(
            title="Select optimization output root",
            initialdir=self.output_var.get() or PROJECT_ROOT,
        )
        if path:
            self.output_var.set(path)

    def _append_log(self, message):
        self.log.config(state=self.tk.NORMAL)
        self.log.insert(self.tk.END, message + "\n")
        self.log.see(self.tk.END)
        self.log.config(state=self.tk.DISABLED)

    def _progress(self, key, status, message):
        self.root.after(0, self._append_log, message)

    def _start(self):
        from tkinter import messagebox
        selected = tuple(key for key, _ in ALGORITHMS if self.algorithm_vars[key].get())
        try:
            request = RunRequest(
                model=Path(self.model_var.get()),
                train=Path(self.train_var.get()),
                validation=Path(self.validation_var.get()),
                test=Path(self.test_var.get()) if self.test_var.get().strip() else None,
                algorithms=selected,
                output_root=Path(self.output_var.get()),
                permutations=self.permutations_var.get(),
                test_permutations=self.test_permutations_var.get(),
                seed=self.seed_var.get(),
                install_sidecars=self.install_var.get(),
                apply_algorithm_defaults=self.apply_defaults_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.run_button.config(state=self.tk.DISABLED, text="Running...")
        self.log.config(state=self.tk.NORMAL)
        self.log.delete("1.0", self.tk.END)
        self.log.config(state=self.tk.DISABLED)

        def worker():
            try:
                manifest = run_optimization(request, self._progress)
                self.root.after(0, self._finish, manifest, None)
            except Exception as exc:
                self.root.after(0, self._finish, None, (exc, traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def _finish(self, manifest, error):
        from tkinter import messagebox
        self.run_button.config(state=self.tk.NORMAL, text="Run Optimization")
        if error:
            self._append_log(error[1])
            messagebox.showerror("Optimization failed", str(error[0]))
            return
        self.last_run = Path(manifest["run_directory"])
        self.run_path_var.set(str(self.last_run))
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        for row in manifest["summary"]:
            fmt = lambda value: "-" if value is None else f"{value * 100:.2f}%"
            nll = "-" if row["validation_nll"] is None else f"{row['validation_nll']:.4f}"
            self.metrics_tree.insert("", self.tk.END, values=(row["algorithm"], fmt(row["validation_accuracy"]), nll, fmt(row["test_accuracy"])))
        png_paths = [Path(path) for path in manifest["charts"] if str(path).lower().endswith(".png")]
        self.chart_paths = {path.stem.replace("_", " ").title(): path for path in png_paths}
        names = list(self.chart_paths)
        self.chart_selector.config(values=names)
        if names:
            self.chart_var.set(names[0])
            self._show_selected_chart()
        messagebox.showinfo("Optimization complete", f"Results saved to:\n{self.last_run}")

    def _open_run(self):
        if self.last_run and self.last_run.is_dir():
            os.startfile(self.last_run)

    def _view_metrics(self):
        from tkinter import messagebox
        selection = self.metrics_tree.selection()
        if not self.last_run:
            messagebox.showinfo("Metrics", "Complete an optimization run first.")
            return
        if not selection:
            messagebox.showinfo("Metrics", "Select an algorithm row first.")
            return
        display = self.metrics_tree.item(selection[0], "values")[0]
        key = next((key for key, name in ALGORITHMS if name == display), None)
        path = self.last_run / str(key) / "evaluation_report.json"
        if not path.is_file():
            messagebox.showerror("Metrics", f"Metrics report not found:\n{path}")
            return
        window = self.tk.Toplevel(self.root)
        window.title(f"{display} Metrics")
        window.geometry("840x680")
        frame = self.tk.Frame(window, bg="#ffffff", padx=10, pady=10)
        frame.pack(fill=self.tk.BOTH, expand=True)
        vertical = self.ttk.Scrollbar(frame, orient=self.tk.VERTICAL)
        horizontal = self.ttk.Scrollbar(frame, orient=self.tk.HORIZONTAL)
        text = self.tk.Text(
            frame,
            wrap=self.tk.NONE,
            font=("Consolas", 9),
            bg="#111827",
            fg="#d1d5db",
            yscrollcommand=vertical.set,
            xscrollcommand=horizontal.set,
        )
        vertical.config(command=text.yview)
        horizontal.config(command=text.xview)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        text.insert("1.0", path.read_text(encoding="utf-8"))
        text.config(state=self.tk.DISABLED)

    def _show_selected_chart(self, event=None):
        path = self.chart_paths.get(self.chart_var.get())
        if not path or not path.is_file():
            return
        from PIL import Image, ImageTk
        image = Image.open(path)
        image.thumbnail((980, 430))
        self.chart_image = ImageTk.PhotoImage(image)
        self.chart_label.config(image=self.chart_image, text="")

    def _export_run(self):
        from tkinter import filedialog, messagebox
        if not self.last_run:
            messagebox.showinfo("Export", "Complete an optimization run first.")
            return
        destination = filedialog.askdirectory(title="Export complete run to")
        if not destination:
            return
        target = Path(destination) / self.last_run.parent.name / self.last_run.name
        if target.exists():
            messagebox.showerror("Export", f"Destination already exists:\n{target}")
            return
        shutil.copytree(self.last_run, target)
        messagebox.showinfo("Export complete", f"Exported to:\n{target}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run without opening the GUI")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--algorithms", default=",".join(key for key, _ in ALGORITHMS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--test-permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-install-sidecars", action="store_true")
    parser.add_argument("--no-apply-algorithm-defaults", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.run:
        if not args.model:
            raise ValueError("--model is required with --run")
        algorithms = tuple(item.strip() for item in args.algorithms.split(",") if item.strip())
        manifest = run_optimization(
            RunRequest(
                model=args.model,
                train=args.train,
                validation=args.validation,
                test=args.test,
                algorithms=algorithms,
                output_root=args.output_root,
                permutations=args.permutations,
                test_permutations=args.test_permutations,
                seed=args.seed,
                install_sidecars=not args.no_install_sidecars,
                apply_algorithm_defaults=not args.no_apply_algorithm_defaults,
            ),
            lambda _key, _status, message: print(message, flush=True),
        )
        print(manifest["run_directory"])
        return 0
    import tkinter as tk
    root = tk.Tk()
    OptimizationApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
