import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
import json
import os
from pathlib import Path
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.bayesian.artifacts import (
    ACTIVE_CONFIG_FILE,
    DATASET_FILE,
    build_generated_dataset_path,
    build_run_paths,
    write_json,
)
from algorithms.bayesian.dataset_sequences import (
    generate_random_sequences,
    scan_local_features,
)

_root_dir = os.fspath(PROJECT_ROOT)

# Define the standard shapes and features
SHAPES = ["cube", "sphere", "cylinder", "cone", "square_pyramid"]
FEATURES = ["planar", "single_curvature", "double_curvature", 
            "straight_edge", "circular_rim", "multi_face_vertex", "sharp_apex"]

class OptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bayesian Shape Optimizer")
        self.root.geometry("800x600")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet_model = None
        self.val_transform = None
        self.class_names = []
        
        self.cap = None
        self.running = False
        self.current_sequence = []
        self.dataset_file = DATASET_FILE
        self.dataset_scan = None
        self.editable_features_by_shape = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- TAB 1: Data Collection ---
        tab_data = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_data, text="Data Collection")
        
        # Model Selector
        tk.Label(tab_data, text="Select ResNet Weights (.pth):", bg="#ffffff", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        model_frame = tk.Frame(tab_data, bg="#ffffff")
        model_frame.pack(fill=tk.X, pady=2)
        self.model_var = tk.StringVar()
        tk.Entry(model_frame, textvariable=self.model_var, state="readonly", bg="#e9ecef").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(model_frame, text="Browse", command=self.browse_weights).pack(side=tk.RIGHT, padx=5)
        
        # Camera
        cam_frame = tk.Frame(tab_data, bg="#ffffff")
        cam_frame.pack(fill=tk.X, pady=5)
        self.btn_cam = tk.Button(cam_frame, text="Start Camera", command=self.toggle_camera, bg="#007bff", fg="white")
        self.btn_cam.pack(side=tk.LEFT, padx=5)
        
        self.lbl_video = tk.Label(tab_data, bg="#000000", width=400, height=300)
        self.lbl_video.pack(pady=5)
        
        # Current Prediction
        self.pred_var = tk.StringVar(value="Waiting...")
        tk.Label(tab_data, textvariable=self.pred_var, font=("Segoe UI", 12, "bold"), fg="#28a745", bg="#ffffff").pack(pady=5)
        
        # Sequence Recording
        seq_frame = tk.Frame(tab_data, bg="#f8f9fa", bd=1, relief=tk.SOLID)
        seq_frame.pack(fill=tk.X, pady=5, padx=5)
        
        tk.Label(seq_frame, text="Ground Truth Shape:", bg="#f8f9fa").grid(row=0, column=0, padx=5, pady=5)
        self.gt_shape_var = tk.StringVar(value="cube")
        ttk.Combobox(seq_frame, textvariable=self.gt_shape_var, values=SHAPES, state="readonly").grid(row=0, column=1, padx=5, pady=5)
        
        tk.Button(seq_frame, text="Record Current Feature", command=self.record_feature, bg="#ffc107").grid(row=0, column=2, padx=5, pady=5)
        tk.Button(seq_frame, text="Save Sequence", command=self.save_sequence, bg="#28a745", fg="white").grid(row=0, column=3, padx=5, pady=5)
        
        self.lbl_seq = tk.Label(seq_frame, text="Sequence: []", bg="#f8f9fa", font=("Consolas", 9))
        self.lbl_seq.grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=5, pady=5)
        
        # --- TAB 2: Optimization ---
        tab_opt = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_opt, text="Optimization")
        
        tk.Label(tab_opt, text="Optimize Bayesian Likelihoods", font=("Segoe UI", 14, "bold"), bg="#ffffff").pack(pady=10)

        dataset_select = tk.Frame(tab_opt, bg="#ffffff")
        dataset_select.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Label(dataset_select, text="Touch Sequence JSON:", bg="#ffffff", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        dataset_path_row = tk.Frame(dataset_select, bg="#ffffff")
        dataset_path_row.pack(fill=tk.X, pady=2)
        self.optimizer_dataset_var = tk.StringVar(value=str(self.dataset_file))
        tk.Entry(dataset_path_row, textvariable=self.optimizer_dataset_var, state="readonly", bg="#e9ecef").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(dataset_path_row, text="Browse", command=self.browse_optimizer_dataset).pack(side=tk.RIGHT, padx=(5, 0))
        
        self.txt_opt_logs = tk.Text(tab_opt, height=15, bg="#f8f9fa", font=("Consolas", 9))
        self.txt_opt_logs.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Button(tab_opt, text="Run Gradient Descent Optimization", command=self.run_optimization, bg="#007bff", fg="white", font=("Segoe UI", 11, "bold"), pady=10).pack(fill=tk.X, padx=10, pady=10)

        # --- TAB 3: Dataset Browser and Random Sequence Generator ---
        tab_dataset = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_dataset, text="Dataset Generator")

        tk.Label(tab_dataset, text="Metadata Feature Scanner", font=("Segoe UI", 14, "bold"), bg="#ffffff").pack(anchor=tk.W, padx=10, pady=(10, 5))
        tk.Label(
            tab_dataset,
            text="Reads label and custom_fields.local_feature from metadata JSON files.",
            bg="#ffffff",
            fg="#6c757d",
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, padx=10, pady=(0, 5))

        source_row = tk.Frame(tab_dataset, bg="#ffffff")
        source_row.pack(fill=tk.X, padx=10, pady=4)
        self.dataset_root_var = tk.StringVar(value=os.path.join(_root_dir, "dataset", "GelSight"))
        tk.Entry(source_row, textvariable=self.dataset_root_var, state="readonly", bg="#e9ecef").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(source_row, text="Browse", command=self.browse_dataset_directory).pack(side=tk.LEFT, padx=(5, 0))
        self.btn_scan_dataset = tk.Button(source_row, text="Scan", command=self.scan_dataset, bg="#007bff", fg="white")
        self.btn_scan_dataset.pack(side=tk.LEFT, padx=(5, 0))

        self.dataset_scan_status_var = tk.StringVar(value="Select a dataset directory, then scan metadata.")
        tk.Label(tab_dataset, textvariable=self.dataset_scan_status_var, bg="#ffffff", fg="#495057", anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=10, pady=(2, 5))

        self.tree_dataset_features = ttk.Treeview(
            tab_dataset,
            columns=("Shape", "Records", "Features"),
            show="headings",
            height=7,
        )
        self.tree_dataset_features.heading("Shape", text="Object Class")
        self.tree_dataset_features.heading("Records", text="Metadata")
        self.tree_dataset_features.heading("Features", text="Unique Local Features")
        self.tree_dataset_features.column("Shape", width=130, stretch=False)
        self.tree_dataset_features.column("Records", width=80, anchor=tk.CENTER, stretch=False)
        self.tree_dataset_features.column("Features", width=480, stretch=True)
        self.tree_dataset_features.pack(fill=tk.X, padx=10, pady=(0, 4))
        self.tree_dataset_features.bind("<Double-1>", self.edit_selected_class_features)

        feature_actions = tk.Frame(tab_dataset, bg="#ffffff")
        feature_actions.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Button(
            feature_actions,
            text="Edit Selected Features...",
            command=self.edit_selected_class_features,
        ).pack(side=tk.LEFT)
        tk.Button(
            feature_actions,
            text="Reset from Metadata",
            command=self.reset_feature_edits,
        ).pack(side=tk.LEFT, padx=(5, 0))

        settings = tk.LabelFrame(tab_dataset, text="Random Sequence Settings", bg="#ffffff", padx=8, pady=6)
        settings.pack(fill=tk.X, padx=10, pady=5)
        self.generated_count_var = tk.IntVar(value=50)
        self.generated_min_length_var = tk.IntVar(value=3)
        self.generated_max_length_var = tk.IntVar(value=8)
        self.generated_seed_var = tk.IntVar(value=42)
        controls = (
            ("Sequences per class", self.generated_count_var, 1, 100000),
            ("Minimum length", self.generated_min_length_var, 1, 100),
            ("Maximum length", self.generated_max_length_var, 1, 100),
            ("Random seed", self.generated_seed_var, 0, 2147483647),
        )
        for column, (label, variable, minimum, maximum) in enumerate(controls):
            tk.Label(settings, text=label, bg="#ffffff").grid(row=0, column=column, padx=5, sticky=tk.W)
            tk.Spinbox(settings, from_=minimum, to=maximum, textvariable=variable, width=12).grid(row=1, column=column, padx=5, pady=(2, 0), sticky=tk.W)

        tk.Button(
            tab_dataset,
            text="Generate Random Touch Sequences",
            command=self.generate_dataset_sequences,
            bg="#28a745",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            pady=7,
        ).pack(fill=tk.X, padx=10, pady=(8, 5))

        self.generated_status_var = tk.StringVar(value="No generated dataset yet.")
        tk.Label(tab_dataset, textvariable=self.generated_status_var, bg="#ffffff", fg="#495057", anchor=tk.W, justify=tk.LEFT, wraplength=740).pack(fill=tk.X, padx=10, pady=(2, 8))

    def set_optimizer_dataset(self, path):
        self.dataset_file = os.path.abspath(os.fspath(path))
        self.optimizer_dataset_var.set(self.dataset_file)

    def browse_optimizer_dataset(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(os.fspath(self.dataset_file)),
            title="Select Touch Sequence Dataset",
            filetypes=(("JSON Dataset", "*.json"), ("All Files", "*.*")),
        )
        if path:
            self.set_optimizer_dataset(path)

    def browse_dataset_directory(self):
        path = filedialog.askdirectory(
            initialdir=self.dataset_root_var.get() or _root_dir,
            title="Select Metadata Dataset Directory",
        )
        if path:
            self.dataset_root_var.set(path)
            self.dataset_scan = None
            self.editable_features_by_shape = {}
            self.refresh_dataset_feature_tree()
            self.dataset_scan_status_var.set("Dataset changed. Click Scan to read metadata.")

    def scan_dataset(self):
        root_path = self.dataset_root_var.get()
        if not root_path:
            messagebox.showerror("Dataset", "Select a dataset directory first.")
            return

        self.btn_scan_dataset.config(state=tk.DISABLED, text="Scanning...")
        self.dataset_scan_status_var.set("Scanning metadata files...")

        def worker():
            try:
                result = scan_local_features(
                    root_path,
                    valid_shapes=SHAPES,
                    valid_features=FEATURES,
                )
                self.root.after(0, lambda: self.finish_dataset_scan(result, None))
            except Exception as exc:
                self.root.after(0, lambda error=exc: self.finish_dataset_scan(None, error))

        threading.Thread(target=worker, daemon=True).start()

    def finish_dataset_scan(self, result, error):
        self.btn_scan_dataset.config(state=tk.NORMAL, text="Scan")
        if error is not None:
            self.dataset_scan = None
            self.dataset_scan_status_var.set(f"Scan failed: {error}")
            messagebox.showerror("Dataset Scan Failed", str(error))
            return

        self.dataset_scan = result
        self.editable_features_by_shape = {
            shape: tuple(features)
            for shape, features in result.features_by_shape.items()
        }
        self.refresh_dataset_feature_tree()

        summary = (
            f"Read {result.metadata_files} metadata files; accepted "
            f"{result.accepted_records}; found {len(result.features_by_shape)} classes."
        )
        warnings = []
        if result.invalid_records:
            warnings.append(f"invalid JSON/metadata: {result.invalid_records}")
        if result.ignored_records:
            warnings.append(f"missing labels: {result.ignored_records}")
        if result.unknown_shapes:
            warnings.append("unknown classes: " + ", ".join(result.unknown_shapes))
        if result.unknown_features:
            warnings.append("unknown features: " + ", ".join(result.unknown_features))
        if warnings:
            summary += " Warnings: " + "; ".join(warnings)
        self.dataset_scan_status_var.set(summary)

    def refresh_dataset_feature_tree(self):
        if not hasattr(self, "tree_dataset_features"):
            return
        selected_shapes = {
            self.tree_dataset_features.item(item, "values")[0]
            for item in self.tree_dataset_features.selection()
            if self.tree_dataset_features.item(item, "values")
        }
        for item in self.tree_dataset_features.get_children():
            self.tree_dataset_features.delete(item)
        for shape, features in sorted(self.editable_features_by_shape.items()):
            records = (
                self.dataset_scan.records_by_shape.get(shape, 0)
                if self.dataset_scan is not None
                else 0
            )
            self.tree_dataset_features.insert(
                "",
                tk.END,
                iid=shape,
                values=(
                    shape,
                    records,
                    ", ".join(features),
                ),
            )
            if shape in selected_shapes:
                self.tree_dataset_features.selection_add(shape)

    def edit_selected_class_features(self, event=None):
        selection = self.tree_dataset_features.selection()
        if not selection:
            messagebox.showinfo("Feature Editor", "Select an object class first.")
            return
        shape = self.tree_dataset_features.item(selection[0], "values")[0]
        current_features = set(self.editable_features_by_shape.get(shape, ()))

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit Features - {shape}")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text=shape.replace("_", " ").title(),
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W, padx=14, pady=(12, 4))
        tk.Label(
            dialog,
            text="Select the local features available for random generation.",
            fg="#6c757d",
        ).pack(anchor=tk.W, padx=14, pady=(0, 8))

        feature_frame = tk.Frame(dialog)
        feature_frame.pack(fill=tk.BOTH, padx=14, pady=4)
        feature_vars = {}
        for index, feature in enumerate(FEATURES):
            variable = tk.BooleanVar(value=feature in current_features)
            feature_vars[feature] = variable
            tk.Checkbutton(
                feature_frame,
                text=feature.replace("_", " ").title(),
                variable=variable,
            ).grid(row=index // 2, column=index % 2, sticky=tk.W, padx=(0, 22), pady=3)

        quick_actions = tk.Frame(dialog)
        quick_actions.pack(fill=tk.X, padx=14, pady=(5, 0))
        tk.Button(
            quick_actions,
            text="Select All",
            command=lambda: [variable.set(True) for variable in feature_vars.values()],
        ).pack(side=tk.LEFT)
        tk.Button(
            quick_actions,
            text="Clear",
            command=lambda: [variable.set(False) for variable in feature_vars.values()],
        ).pack(side=tk.LEFT, padx=(5, 0))

        def save_selection():
            selected = tuple(
                feature for feature in FEATURES if feature_vars[feature].get()
            )
            if not selected:
                messagebox.showerror(
                    "Feature Editor",
                    "Select at least one local feature for this class.",
                    parent=dialog,
                )
                return
            self.editable_features_by_shape[shape] = selected
            self.refresh_dataset_feature_tree()
            self.tree_dataset_features.selection_set(shape)
            self.dataset_scan_status_var.set(
                f"Edited feature vocabulary for {shape}. Random generation will use this selection."
            )
            dialog.destroy()

        dialog_actions = tk.Frame(dialog)
        dialog_actions.pack(fill=tk.X, padx=14, pady=12)
        tk.Button(dialog_actions, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        tk.Button(
            dialog_actions,
            text="Apply",
            command=save_selection,
            bg="#007bff",
            fg="white",
        ).pack(side=tk.RIGHT, padx=(0, 6))

    def reset_feature_edits(self):
        if self.dataset_scan is None:
            messagebox.showinfo("Feature Editor", "Scan a metadata dataset first.")
            return
        self.editable_features_by_shape = {
            shape: tuple(features)
            for shape, features in self.dataset_scan.features_by_shape.items()
        }
        self.refresh_dataset_feature_tree()
        self.dataset_scan_status_var.set("Feature selections restored from scanned metadata.")

    def generate_dataset_sequences(self):
        if self.dataset_scan is None:
            messagebox.showerror("Dataset", "Scan a metadata dataset first.")
            return

        try:
            generated = generate_random_sequences(
                self.editable_features_by_shape,
                sequences_per_shape=int(self.generated_count_var.get()),
                minimum_length=int(self.generated_min_length_var.get()),
                maximum_length=int(self.generated_max_length_var.get()),
                seed=int(self.generated_seed_var.get()),
            )
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Generation Settings", str(exc))
            return

        if not generated:
            messagebox.showerror("Dataset", "No valid class features were available to generate sequences.")
            return

        output_path = build_generated_dataset_path()
        write_json(output_path, generated)
        self.set_optimizer_dataset(output_path)
        self.generated_status_var.set(
            f"Generated {len(generated)} sequences and selected them for optimization: "
            f"{output_path}"
        )
        messagebox.showinfo(
            "Dataset Generated",
            f"Generated {len(generated)} random touch sequences.\n\nSaved to:\n{output_path}",
        )

    def browse_weights(self):
        path = filedialog.askopenfilename(initialdir=_root_dir, filetypes=(("PyTorch Model", "*.pth"),))
        if path:
            self.model_var.set(path)
            self.load_model(path)
            
    def load_model(self, path):
        try:
            base, _ = os.path.splitext(path)
            labels_path = base + ".txt"
            if os.path.exists(labels_path):
                with open(labels_path, "r") as f:
                    self.class_names = [l.strip() for l in f.readlines() if l.strip()]
            
            num_classes = max(len(self.class_names), 1)
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            model.to(self.device)
            self.resnet_model = model
            self.val_transform = T.Compose([
                T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            messagebox.showinfo("Success", "ResNet Model Loaded!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def toggle_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.btn_cam.config(text="Stop Camera", bg="#dc3545")
            threading.Thread(target=self.video_loop, daemon=True).start()
        else:
            self.running = False
            if self.cap:
                self.cap.release()
            self.btn_cam.config(text="Start Camera", bg="#007bff")
            
    def video_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: continue
            
            # Predict
            pred_class = "Waiting..."
            if self.resnet_model:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    tensor = self.val_transform(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits = self.resnet_model(tensor)
                        probs = torch.nn.functional.softmax(logits, dim=1)[0]
                        idx = torch.argmax(probs).item()
                        if idx < len(self.class_names):
                            pred_class = self.class_names[idx]
                except:
                    pass
            
            self.current_feature = pred_class
            self.root.after(0, lambda p=pred_class: self.pred_var.set(f"Detected: {p}"))
            
            # Display
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(frame, (400, 300)), cv2.COLOR_BGR2RGB)))
            self.root.after(0, lambda img=img_tk: self.update_image(img))
            time.sleep(0.05)
            
    def update_image(self, img):
        self.lbl_video.config(image=img)
        self.lbl_video.image = img
        
    def record_feature(self):
        feat = getattr(self, "current_feature", "").lower()
        if feat == "multiface_vertex":
            feat = "multi_face_vertex"
            
        if feat and feat in FEATURES:
            self.current_sequence.append(feat)
            self.lbl_seq.config(text=f"Sequence: {self.current_sequence}")
        else:
            messagebox.showwarning("Warning", f"Valid local feature not detected yet. Detected: '{feat}'")
            
    def save_sequence(self):
        if not self.current_sequence:
            return
        
        gt = self.gt_shape_var.get()
        DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        if DATASET_FILE.exists():
            with DATASET_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                
        data.append({"shape": gt, "sequence": self.current_sequence})
        
        write_json(DATASET_FILE, data)
        self.set_optimizer_dataset(DATASET_FILE)
            
        self.current_sequence = []
        self.lbl_seq.config(text="Sequence: []")
        messagebox.showinfo("Saved", f"Sequence for {gt} saved to:\n{DATASET_FILE}")

    def log_opt(self, msg):
        self.txt_opt_logs.insert(tk.END, msg + "\n")
        self.txt_opt_logs.see(tk.END)
        self.root.update()

    def run_optimization(self):
        dataset_file = os.path.abspath(os.fspath(self.dataset_file))
        if not os.path.exists(dataset_file):
            messagebox.showerror("Error", "No dataset found! Record sequences first.")
            return
            
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Invalid Dataset", f"Could not read dataset:\n{exc}")
            return
            
        if not isinstance(data, list) or not data:
            messagebox.showerror("Error", "Dataset must be a non-empty JSON array.")
            return
            
        self.txt_opt_logs.delete(1.0, tk.END)
        self.log_opt(f"Loaded {len(data)} touch sequences.")
        self.log_opt(f"Dataset: {dataset_file}")
        self.log_opt("Initializing PyTorch Optimizer (Gradient Descent)...")
        
        # We want to learn P(Feature | Shape).
        # We parameterize it as logits `W` of shape [num_features, num_shapes]
        # such that Softmax(W) across features gives a valid probability distribution.
        
        W = nn.Parameter(torch.zeros(len(FEATURES), len(SHAPES)))
        optimizer = torch.optim.Adam([W], lr=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # Map labels to indices
        shape2idx = {s: i for i, s in enumerate(SHAPES)}
        feat2idx = {f: i for i, f in enumerate(FEATURES)}

        training_samples = []
        validation_errors = []
        for index, item in enumerate(data, start=1):
            shape = item.get("shape") if isinstance(item, dict) else None
            sequence = item.get("sequence") if isinstance(item, dict) else None
            if shape not in shape2idx:
                validation_errors.append(f"Item {index}: unknown shape {shape!r}")
                continue
            if not isinstance(sequence, list) or not sequence:
                validation_errors.append(f"Item {index}: sequence must be a non-empty list")
                continue
            unknown_features = sorted(
                {repr(feature) for feature in sequence if feature not in feat2idx}
            )
            if unknown_features:
                validation_errors.append(
                    f"Item {index}: unknown features {', '.join(unknown_features)}"
                )
                continue
            training_samples.append(
                (shape2idx[shape], [feat2idx[feature] for feature in sequence])
            )

        if validation_errors:
            message = "Dataset validation failed:\n" + "\n".join(validation_errors[:10])
            if len(validation_errors) > 10:
                message += f"\n...and {len(validation_errors) - 10} more errors."
            self.log_opt(message)
            messagebox.showerror("Invalid Dataset", message)
            return

        run_paths = build_run_paths()
        write_json(run_paths.dataset_snapshot, data)
        self.log_opt(f"Dataset snapshot: {run_paths.dataset_snapshot}")
        self.log_opt(
            f"Training samples: {len(training_samples)}. Reported loss is mean cross-entropy."
        )
        
        epochs = 1000
        for epoch in range(epochs):
            optimizer.zero_grad()
            sample_losses = []
            correct_predictions = 0

            # P(feature | shape) is shared by every sequence in this epoch.
            log_probs = torch.nn.functional.log_softmax(W, dim=0)
            for gt_idx, seq in training_samples:
                # Naive Bayes: P(Shape | Seq) propto P(Shape) * prod P(Feat | Shape)
                # In log space: log P(Shape | Seq) = log P(Shape) + sum log P(Feat | Shape)
                # P(Feat | Shape) = Softmax(W, dim=0) -> log P(Feat | Shape) = LogSoftmax(W, dim=0)

                seq_log_probs = log_probs[seq, :].sum(dim=0)

                # We assume flat prior P(Shape), so seq_log_probs are the unnormalized logits for the shapes
                sample_losses.append(
                    criterion(seq_log_probs.unsqueeze(0), torch.tensor([gt_idx]))
                )
                correct_predictions += int(seq_log_probs.argmax().item() == gt_idx)

            loss = torch.stack(sample_losses).mean()
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                accuracy = correct_predictions / len(training_samples)
                self.log_opt(
                    f"Epoch {epoch+1}/{epochs} - Mean CE: {loss.item():.4f} "
                    f"- Training Accuracy: {accuracy * 100:.1f}%"
                )
                
        self.log_opt("Optimization complete! Extracting probabilities...")
        
        final_probs = torch.nn.functional.softmax(W, dim=0).detach().numpy()

        # Clamp tiny values, then renormalize each shape column so every column
        # remains a valid P(feature | shape) distribution.
        final_probs = final_probs.clip(min=0.01)
        final_probs = final_probs / final_probs.sum(axis=0, keepdims=True)
        
        # Format shared by the dated result and active runtime config.
        new_config = {}
        for i, feat in enumerate(FEATURES):
            new_config[feat] = {}
            for j, shape in enumerate(SHAPES):
                new_config[feat][shape] = float(final_probs[i, j])

        write_json(run_paths.optimized_config, new_config)
        write_json(ACTIVE_CONFIG_FILE, new_config)

        self.log_opt(f"Dated config: {run_paths.optimized_config}")
        self.log_opt(f"Active runtime config updated: {ACTIVE_CONFIG_FILE}")
        messagebox.showinfo(
            "Success",
            "Optimization complete.\n\n"
            f"Dated result:\n{run_paths.optimized_config}\n\n"
            f"Active config:\n{ACTIVE_CONFIG_FILE}",
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizerApp(root)
    root.mainloop()
