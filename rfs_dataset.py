"""UI for generating and splitting synthetic hard-label RFS touch trials."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from algorithms.bayesian.dataset_sequences import scan_local_features
from algorithms.rfs.config import load_rfs_config
from algorithms.rfs.dataset_generation import generate_and_split_sequences
from algorithms.rfs.factory import DEFAULT_CONFIG_PATH
from algorithms.rfs.trials import TRIAL_SPLITS, save_trials, trial_file


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "GelSight"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "trials"

COLORS = {
    "background": "#f4f6f8",
    "surface": "#ffffff",
    "border": "#d7dde3",
    "text": "#18222d",
    "muted": "#66727f",
    "nav": "#17324a",
    "primary": "#1473e6",
    "primary_dark": "#0b5fc2",
    "success": "#218739",
    "danger": "#c9363e",
}


class RFSDatasetGeneratorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TouchLab VTS - RFS Dataset Generator")
        self.root.geometry("1180x790")
        self.root.minsize(980, 680)
        self.root.configure(bg=COLORS["background"])

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._configure_styles()

        self.config_path_var = tk.StringVar(value=str(DEFAULT_CONFIG_PATH.resolve()))
        default_dataset = DEFAULT_DATASET_ROOT if DEFAULT_DATASET_ROOT.is_dir() else PROJECT_ROOT / "dataset"
        self.dataset_root_var = tk.StringVar(value=str(default_dataset.resolve()))
        self.output_root_var = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT.resolve()))
        self.scan_status_var = tk.StringVar(value="Dataset not scanned")
        self.output_status_var = tk.StringVar(value="No RFS files generated")
        self.total_status_var = tk.StringVar(value="Ready")

        self.sequence_count_var = tk.IntVar(value=50)
        self.minimum_length_var = tk.IntVar(value=3)
        self.maximum_length_var = tk.IntVar(value=8)
        self.seed_var = tk.IntVar(value=42)
        self.train_ratio_var = tk.DoubleVar(value=70.0)
        self.validation_ratio_var = tk.DoubleVar(value=15.0)
        self.test_ratio_var = tk.DoubleVar(value=15.0)

        self.rfs_config = load_rfs_config(DEFAULT_CONFIG_PATH)
        self.features = tuple(self.rfs_config.features)
        self.shapes = tuple(self.rfs_config.shapes)
        self.dataset_scan = None
        self.editable_features_by_shape = {}
        self.generated_trials = []
        self.generated_splits = {split: [] for split in TRIAL_SPLITS}

        self._build_ui()

    def _configure_styles(self) -> None:
        self.style.configure(
            "Treeview",
            background=COLORS["surface"],
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            rowheight=29,
            bordercolor=COLORS["border"],
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Treeview.Heading",
            background="#e9eef3",
            foreground=COLORS["text"],
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
        )
        self.style.map("Treeview", background=[("selected", "#d8eaff")])
        self.style.configure(
            "TCombobox",
            fieldbackground=COLORS["surface"],
            background=COLORS["surface"],
            foreground=COLORS["text"],
            padding=5,
        )

    def _button(self, parent, text, command, *, kind="secondary"):
        palette = {
            "primary": (COLORS["primary"], "white", COLORS["primary_dark"]),
            "success": (COLORS["success"], "white", "#176a2b"),
            "danger": (COLORS["danger"], "white", "#a52b32"),
            "secondary": ("#e9eef3", COLORS["text"], "#dce3e9"),
        }
        background, foreground, active = palette[kind]
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=active,
            activeforeground=foreground,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )

    def _build_ui(self) -> None:
        nav = tk.Frame(self.root, bg=COLORS["nav"], height=64)
        nav.pack(fill=tk.X)
        nav.pack_propagate(False)
        tk.Label(
            nav,
            text="TOUCHLAB VTS",
            bg=COLORS["nav"],
            fg="white",
            font=("Segoe UI", 16, "bold"),
        ).pack(side=tk.LEFT, padx=(22, 24))
        tk.Label(
            nav,
            text="RFS Dataset Generator",
            bg=COLORS["nav"],
            fg="#bed0df",
            font=("Segoe UI", 11, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            nav,
            textvariable=self.total_status_var,
            bg=COLORS["nav"],
            fg="#bed0df",
            font=("Segoe UI", 9),
        ).pack(side=tk.RIGHT, padx=20)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        self.setup_tab = tk.Frame(notebook, bg=COLORS["background"])
        self.preview_tab = tk.Frame(notebook, bg=COLORS["background"])
        notebook.add(self.setup_tab, text="Generate and Split")
        notebook.add(self.preview_tab, text="Generated Trials")
        self._build_setup_tab()
        self._build_preview_tab()

    def _section(self, parent, title, row, *, weight=False):
        frame = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        frame.grid(row=row, column=0, sticky="nsew" if weight else "ew", pady=(0, 12))
        frame.grid_columnconfigure(0, weight=1)
        if weight:
            frame.grid_rowconfigure(1, weight=1)
        tk.Label(
            frame,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 12, "bold"),
            anchor=tk.W,
            padx=14,
            pady=11,
        ).grid(row=0, column=0, sticky="ew")
        return frame

    def _path_row(self, parent, row, label, variable, command, button_text="Browse"):
        container = tk.Frame(parent, bg=COLORS["surface"], padx=14)
        container.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        container.grid_columnconfigure(1, weight=1)
        tk.Label(
            container,
            text=label,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            width=16,
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="w")
        tk.Entry(
            container,
            textvariable=variable,
            state="readonly",
            readonlybackground="#f7f9fb",
            fg=COLORS["text"],
            relief=tk.FLAT,
            font=("Segoe UI", 9),
        ).grid(row=0, column=1, sticky="ew", padx=8, ipady=7)
        self._button(container, button_text, command).grid(row=0, column=2)

    def _build_setup_tab(self) -> None:
        tab = self.setup_tab
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        source = self._section(tab, "Source", 0)
        self._path_row(source, 1, "Metadata dataset", self.dataset_root_var, self.browse_dataset)
        self._path_row(source, 2, "RFS config", self.config_path_var, self.browse_config)
        self._path_row(source, 3, "Output folder", self.output_root_var, self.browse_output)
        source_actions = tk.Frame(source, bg=COLORS["surface"], padx=14)
        source_actions.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        self.scan_button = self._button(
            source_actions, "Scan Metadata", self.scan_dataset, kind="primary"
        )
        self.scan_button.pack(side=tk.LEFT)
        tk.Label(
            source_actions,
            textvariable=self.scan_status_var,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=12)

        body = tk.Frame(tab, bg=COLORS["background"])
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)

        vocabulary = tk.Frame(
            body,
            bg=COLORS["surface"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        vocabulary.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        vocabulary.grid_rowconfigure(1, weight=1)
        vocabulary.grid_columnconfigure(0, weight=1)
        vocab_header = tk.Frame(vocabulary, bg=COLORS["surface"], padx=14, pady=10)
        vocab_header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            vocab_header,
            text="Class Feature Vocabulary",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT)
        self._button(vocab_header, "Reset", self.reset_feature_edits).pack(side=tk.RIGHT)
        self._button(vocab_header, "Edit Selected", self.edit_selected_features).pack(
            side=tk.RIGHT, padx=(0, 8)
        )
        self.feature_tree = ttk.Treeview(
            vocabulary,
            columns=("shape", "records", "features"),
            show="headings",
            selectmode="browse",
        )
        self.feature_tree.heading("shape", text="Object class")
        self.feature_tree.heading("records", text="Metadata")
        self.feature_tree.heading("features", text="Unique local features")
        self.feature_tree.column("shape", width=150, stretch=False)
        self.feature_tree.column("records", width=85, anchor=tk.CENTER, stretch=False)
        self.feature_tree.column("features", width=430, stretch=True)
        feature_scroll = ttk.Scrollbar(
            vocabulary, orient=tk.VERTICAL, command=self.feature_tree.yview
        )
        self.feature_tree.configure(yscrollcommand=feature_scroll.set)
        self.feature_tree.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(0, 12))
        feature_scroll.grid(row=1, column=1, sticky="ns", pady=(0, 12), padx=(0, 12))
        self.feature_tree.bind("<Double-1>", self.edit_selected_features)

        settings = tk.Frame(
            body,
            bg=COLORS["surface"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        settings.grid(row=0, column=1, sticky="nsew")
        settings.grid_columnconfigure(0, weight=1)
        tk.Label(
            settings,
            text="Generation",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 12, "bold"),
            anchor=tk.W,
            padx=14,
            pady=11,
        ).grid(row=0, column=0, sticky="ew")

        controls = tk.Frame(settings, bg=COLORS["surface"], padx=14)
        controls.grid(row=1, column=0, sticky="ew")
        controls.grid_columnconfigure(1, weight=1)
        generation_fields = (
            ("Sequences per class", self.sequence_count_var, 1, 100000),
            ("Minimum length", self.minimum_length_var, 1, 100),
            ("Maximum length", self.maximum_length_var, 1, 100),
            ("Random seed", self.seed_var, 0, 2147483647),
        )
        for row, (label, variable, minimum, maximum) in enumerate(generation_fields):
            tk.Label(
                controls, text=label, bg=COLORS["surface"], fg=COLORS["muted"]
            ).grid(row=row, column=0, sticky="w", pady=5)
            tk.Spinbox(
                controls,
                from_=minimum,
                to=maximum,
                textvariable=variable,
                relief=tk.FLAT,
                bg="#f7f9fb",
                buttonbackground="#e9eef3",
                font=("Segoe UI", 10),
            ).grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=5, ipady=4)

        tk.Label(
            settings,
            text="Split percentages",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W,
            padx=14,
            pady=10,
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ratios = tk.Frame(settings, bg=COLORS["surface"], padx=14)
        ratios.grid(row=3, column=0, sticky="ew")
        ratios.grid_columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate(
            (("Train", self.train_ratio_var), ("Validation", self.validation_ratio_var), ("Test", self.test_ratio_var))
        ):
            tk.Label(ratios, text=label, bg=COLORS["surface"], fg=COLORS["muted"]).grid(
                row=row, column=0, sticky="w", pady=5
            )
            tk.Spinbox(
                ratios,
                from_=0,
                to=100,
                increment=1,
                textvariable=variable,
                relief=tk.FLAT,
                bg="#f7f9fb",
                buttonbackground="#e9eef3",
                font=("Segoe UI", 10),
            ).grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=5, ipady=4)

        action = tk.Frame(settings, bg=COLORS["surface"], padx=14, pady=14)
        action.grid(row=4, column=0, sticky="sew")
        self.generate_button = self._button(
            action, "Generate and Split", self.generate_and_split, kind="success"
        )
        self.generate_button.pack(fill=tk.X)
        tk.Label(
            action,
            textvariable=self.output_status_var,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            justify=tk.LEFT,
            wraplength=360,
            font=("Segoe UI", 9),
        ).pack(fill=tk.X, pady=(10, 0))

    def _build_preview_tab(self) -> None:
        tab = self.preview_tab
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        header = tk.Frame(tab, bg=COLORS["surface"], padx=14, pady=12)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.preview_summary_var = tk.StringVar(value="No generated trials")
        tk.Label(
            header,
            textvariable=self.preview_summary_var,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 11, "bold"),
        ).pack(side=tk.LEFT)
        self._button(header, "Open Output Folder", self.open_output_folder).pack(side=tk.RIGHT)

        table = tk.Frame(
            tab,
            bg=COLORS["surface"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        table.grid(row=1, column=0, sticky="nsew")
        table.grid_rowconfigure(0, weight=1)
        table.grid_columnconfigure(0, weight=1)
        self.preview_tree = ttk.Treeview(
            table,
            columns=("split", "shape", "length", "sequence"),
            show="headings",
        )
        self.preview_tree.heading("split", text="Split")
        self.preview_tree.heading("shape", text="Object class")
        self.preview_tree.heading("length", text="Length")
        self.preview_tree.heading("sequence", text="Touch sequence")
        self.preview_tree.column("split", width=105, stretch=False)
        self.preview_tree.column("shape", width=165, stretch=False)
        self.preview_tree.column("length", width=70, anchor=tk.CENTER, stretch=False)
        self.preview_tree.column("sequence", width=700, stretch=True)
        scroll = ttk.Scrollbar(table, orient=tk.VERTICAL, command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=scroll.set)
        self.preview_tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def browse_dataset(self) -> None:
        path = filedialog.askdirectory(
            title="Select metadata dataset",
            initialdir=self.dataset_root_var.get(),
        )
        if path:
            self.dataset_root_var.set(path)
            self.dataset_scan = None
            self.editable_features_by_shape = {}
            self.scan_status_var.set("Dataset changed; scan metadata again")
            self.refresh_feature_tree()

    def browse_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select RFS configuration",
            initialdir=str(Path(self.config_path_var.get()).parent),
            filetypes=(("RFS JSON", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            config = load_rfs_config(path)
        except Exception as exc:
            messagebox.showerror("RFS Config", str(exc))
            return
        self.config_path_var.set(path)
        self.rfs_config = config
        self.features = tuple(config.features)
        self.shapes = tuple(config.shapes)
        self.dataset_scan = None
        self.editable_features_by_shape = {}
        self.scan_status_var.set(f"Loaded config {config.name}; scan metadata again")
        self.refresh_feature_tree()

    def browse_output(self) -> None:
        path = filedialog.askdirectory(
            title="Select RFS output folder",
            initialdir=self.output_root_var.get(),
        )
        if path:
            self.output_root_var.set(path)

    def scan_dataset(self) -> None:
        self.scan_button.configure(state=tk.DISABLED, text="Scanning...")
        self.total_status_var.set("Scanning metadata")

        def worker():
            try:
                result = scan_local_features(
                    self.dataset_root_var.get(),
                    valid_shapes=self.shapes,
                    valid_features=self.features,
                    feature_aliases=self.rfs_config.aliases,
                )
                self.root.after(0, lambda: self.finish_scan(result, None))
            except Exception as exc:
                self.root.after(0, lambda error=exc: self.finish_scan(None, error))

        threading.Thread(target=worker, daemon=True).start()

    def finish_scan(self, result, error) -> None:
        self.scan_button.configure(state=tk.NORMAL, text="Scan Metadata")
        if error is not None:
            self.dataset_scan = None
            self.scan_status_var.set(f"Scan failed: {error}")
            self.total_status_var.set("Scan failed")
            messagebox.showerror("Metadata Scan", str(error))
            return
        self.dataset_scan = result
        self.editable_features_by_shape = {
            shape: tuple(result.features_by_shape.get(shape, ()))
            for shape in self.shapes
        }
        self.refresh_feature_tree()
        self.scan_status_var.set(
            f"{result.accepted_records} accepted records from {result.metadata_files} metadata files"
        )
        self.total_status_var.set("Metadata scan complete")

    def refresh_feature_tree(self) -> None:
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        for shape in self.shapes:
            features = self.editable_features_by_shape.get(shape, ())
            records = (
                self.dataset_scan.records_by_shape.get(shape, 0)
                if self.dataset_scan is not None
                else 0
            )
            self.feature_tree.insert(
                "",
                tk.END,
                iid=shape,
                values=(shape, records, ", ".join(features)),
            )

    def edit_selected_features(self, _event=None) -> None:
        selected = self.feature_tree.selection()
        if not selected:
            messagebox.showinfo("Feature Editor", "Select an object class first.")
            return
        shape = selected[0]
        current = set(self.editable_features_by_shape.get(shape, ()))
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Feature Vocabulary - {shape}")
        dialog.resizable(False, False)
        dialog.configure(bg=COLORS["surface"])
        dialog.transient(self.root)
        dialog.grab_set()
        tk.Label(
            dialog,
            text=shape.replace("_", " ").title(),
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(14, 8))
        feature_frame = tk.Frame(dialog, bg=COLORS["surface"])
        feature_frame.pack(fill=tk.BOTH, padx=16)
        variables = {}
        for index, feature in enumerate(self.features):
            variable = tk.BooleanVar(value=feature in current)
            variables[feature] = variable
            tk.Checkbutton(
                feature_frame,
                text=feature.replace("_", " ").title(),
                variable=variable,
                bg=COLORS["surface"],
                activebackground=COLORS["surface"],
                selectcolor=COLORS["surface"],
                fg=COLORS["text"],
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 24), pady=4)

        def apply_selection():
            chosen = tuple(feature for feature in self.features if variables[feature].get())
            if not chosen:
                messagebox.showerror(
                    "Feature Editor", "Select at least one feature.", parent=dialog
                )
                return
            self.editable_features_by_shape[shape] = chosen
            self.refresh_feature_tree()
            self.feature_tree.selection_set(shape)
            dialog.destroy()

        actions = tk.Frame(dialog, bg=COLORS["surface"], padx=16, pady=14)
        actions.pack(fill=tk.X)
        self._button(actions, "Apply", apply_selection, kind="primary").pack(side=tk.RIGHT)
        self._button(actions, "Cancel", dialog.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    def reset_feature_edits(self) -> None:
        if self.dataset_scan is None:
            return
        self.editable_features_by_shape = {
            shape: tuple(self.dataset_scan.features_by_shape.get(shape, ()))
            for shape in self.shapes
        }
        self.refresh_feature_tree()

    def generate_and_split(self) -> None:
        if self.dataset_scan is None:
            messagebox.showerror("RFS Dataset", "Scan the metadata dataset first.")
            return
        missing = [
            shape
            for shape in self.shapes
            if not self.editable_features_by_shape.get(shape)
        ]
        if missing:
            messagebox.showerror(
                "RFS Dataset",
                "No local features are selected for: " + ", ".join(missing),
            )
            return
        ratios = (
            float(self.train_ratio_var.get()) / 100.0,
            float(self.validation_ratio_var.get()) / 100.0,
            float(self.test_ratio_var.get()) / 100.0,
        )
        if ratios[0] <= 0.0 or ratios[1] <= 0.0:
            messagebox.showerror(
                "RFS Dataset",
                "Train and validation percentages must both be greater than zero.",
            )
            return
        self.generate_button.configure(state=tk.DISABLED, text="Generating...")
        try:
            generated, splits = generate_and_split_sequences(
                self.editable_features_by_shape,
                sequences_per_shape=int(self.sequence_count_var.get()),
                minimum_length=int(self.minimum_length_var.get()),
                maximum_length=int(self.maximum_length_var.get()),
                seed=int(self.seed_var.get()),
                train_ratio=ratios[0],
                validation_ratio=ratios[1],
                test_ratio=ratios[2],
            )
            output_root = Path(self.output_root_var.get()).expanduser().resolve()
            for split in TRIAL_SPLITS:
                save_trials(trial_file(output_root, split), splits[split])
        except Exception as exc:
            self.generate_button.configure(state=tk.NORMAL, text="Generate and Split")
            messagebox.showerror("RFS Dataset", str(exc))
            return

        self.generated_trials = generated
        self.generated_splits = splits
        self.generate_button.configure(state=tk.NORMAL, text="Generate and Split")
        self.refresh_preview()
        counts = {split: len(splits[split]) for split in TRIAL_SPLITS}
        self.output_status_var.set(
            f"Generated {len(generated)} trials. Train {counts['train']}, "
            f"validation {counts['validation']}, test {counts['test']}."
        )
        self.total_status_var.set("RFS split files saved")
        messagebox.showinfo(
            "RFS Dataset",
            "Saved:\n"
            + "\n".join(str(trial_file(output_root, split)) for split in TRIAL_SPLITS),
        )

    def refresh_preview(self) -> None:
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        for split in TRIAL_SPLITS:
            for trial in self.generated_splits.get(split, []):
                self.preview_tree.insert(
                    "",
                    tk.END,
                    values=(
                        split,
                        trial["shape"],
                        len(trial["sequence"]),
                        ", ".join(trial["sequence"]),
                    ),
                )
        split_counts = {
            split: len(self.generated_splits.get(split, []))
            for split in TRIAL_SPLITS
        }
        self.preview_summary_var.set(
            f"{sum(split_counts.values())} trials  |  Train {split_counts['train']}  |  "
            f"Validation {split_counts['validation']}  |  Test {split_counts['test']}"
        )

    def open_output_folder(self) -> None:
        path = Path(self.output_root_var.get()).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)
        else:
            messagebox.showinfo("Output Folder", str(path))


def main() -> None:
    root = tk.Tk()
    RFSDatasetGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
