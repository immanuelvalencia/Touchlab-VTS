import os
import argparse
import shutil
import json
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from tools.dataset_splitting import DatasetRecord, grouped_split, metadata_group_id


def _metadata_files(input_path, recursive=True):
    if recursive:
        return list(input_path.rglob("*metadata.json"))
    return list(input_path.glob("*metadata.json"))


def _sample_label(metadata, category_by):
    if category_by == "local_feature":
        custom_fields = metadata.get("custom_fields", {})
        label = custom_fields.get("local_feature", "")
    else:
        label = metadata.get("label", "")
    label = str(label).strip()
    return label or None


def scan_dataset(
    input_dir,
    *,
    category_by="label",
    feature_suffix="raw.png",
    recursive=True,
    include_video=False,
):
    """Collect exportable image records grouped by object label or local feature."""
    input_path = Path(input_dir)
    label_to_files = {}

    for meta_file in _metadata_files(input_path, recursive=recursive):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if not include_video and meta.get("is_video_sequence", False):
            continue

        label = _sample_label(meta, category_by)
        if not label or label.lower() == "unknown":
            continue

        saved_features = meta.get("saved_features", [])
        target_file = None
        for saved_f in saved_features:
            if saved_f.endswith(feature_suffix):
                target_file = meta_file.parent / saved_f
                break

        if target_file and target_file.exists():
            sensor_prefix = meta.get("sensor", "UnknownSensor")
            label_to_files.setdefault(label, []).append(
                DatasetRecord(
                    source_file=target_file,
                    sensor_prefix=sensor_prefix,
                    group_id=metadata_group_id(meta_file, meta),
                )
            )

    return label_to_files


def process_image_file(src_file, dst_file, augmentation=None):
    if augmentation:
        with Image.open(src_file) as img:
            if augmentation == "rot180":
                img = img.rotate(180)
            elif augmentation == "hflip":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif augmentation == "vflip":
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(dst_file)
    else:
        shutil.copy2(src_file, dst_file)


def validate_output_directory(output_dir):
    """Prevent old split files from surviving a new grouped export."""
    output_path = Path(output_dir)
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise ValueError(
            f"Export destination must be new or empty: {output_path}. "
            "Choose a new --output_dir (or a new output folder in the GUI). "
            "Reusing an export can leave stale frames in different splits."
        )


def export_records(
    label_to_files,
    output_dir,
    *,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    category_by="label",
    feature_suffix="raw.png",
    augmentations=(),
):
    output_path = Path(output_dir)
    validate_output_directory(output_path)
    for split_dir in ["train", "val", "test"]:
        os.makedirs(output_path / split_dir, exist_ok=True)

    split_manifest = {
        "schema_version": 1,
        "category_by": category_by,
        "feature_suffix": feature_suffix,
        "grouping": "acquisition",
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "classes": {},
    }

    exported_count = 0
    futures = []
    with ThreadPoolExecutor() as executor:
        for label, files_list in label_to_files.items():
            splits = grouped_split(
                files_list,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            split_manifest["classes"][label] = {
                split_name: sorted({record.group_id for record in subset})
                for split_name, subset in splits.items()
            }
    
            for split_name, subset in splits.items():
                target_dir = output_path / split_name / label
                os.makedirs(target_dir, exist_ok=True)
    
                for record in subset:
                    src_file = record.source_file
                    sensor_prefix = record.sensor_prefix
                    base_name = f"{sensor_prefix}_{src_file.name}"
                    futures.append(executor.submit(process_image_file, src_file, target_dir / base_name))
    
                    if split_name == "train":
                        name_no_ext, ext = os.path.splitext(src_file.name)
                        for augmentation in augmentations:
                            aug_file = target_dir / f"{sensor_prefix}_{name_no_ext}_{augmentation}{ext}"
                            futures.append(executor.submit(process_image_file, src_file, aug_file, augmentation))

        for future in as_completed(futures):
            future.result()
            exported_count += 1

    labels = sorted(label_to_files.keys())
    with open(output_path / "labels.txt", "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")

    with open(output_path / "split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=4)
        handle.write("\n")

    return exported_count


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Export a grouped visuo-tactile image dataset from metadata."
    )
    parser.add_argument("--input_dir", default="dataset", help="Directory containing metadata files.")
    parser.add_argument("--output_dir", default="ml_dataset", help="New or empty output dataset directory.")
    parser.add_argument(
        "--category",
        choices=("label", "local_feature"),
        default="label",
        help="Class folder source for the export.",
    )
    parser.add_argument(
        "--by-local-feature",
        action="store_true",
        help="Shortcut for --category local_feature.",
    )
    parser.add_argument("--feature-suffix", default="raw.png", help="Saved image suffix to export.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan subfolders.")
    parser.add_argument("--include-video", action="store_true", help="Include video-sequence frames.")
    parser.add_argument("--train", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Grouped split seed.")
    parser.add_argument("--augment-rot180", action="store_true", help="Add 180-degree rotations to train.")
    parser.add_argument("--augment-hflip", action="store_true", help="Add horizontal flips to train.")
    parser.add_argument("--augment-vflip", action="store_true", help="Add vertical flips to train.")
    return parser.parse_args(argv)


def run_cli(argv):
    args = parse_args(argv)
    category_by = "local_feature" if args.by_local_feature else args.category

    label_to_files = scan_dataset(
        args.input_dir,
        category_by=category_by,
        feature_suffix=args.feature_suffix,
        recursive=not args.no_recursive,
        include_video=args.include_video,
    )
    if not label_to_files:
        print("No valid files found for export.")
        return 1

    augmentations = []
    if args.augment_rot180:
        augmentations.append("rot180")
    if args.augment_hflip:
        augmentations.append("hflip")
    if args.augment_vflip:
        augmentations.append("vflip")

    exported_count = export_records(
        label_to_files,
        args.output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        category_by=category_by,
        feature_suffix=args.feature_suffix,
        augmentations=augmentations,
    )
    print(
        f"Exported {exported_count} files across {len(label_to_files)} "
        f"{category_by.replace('_', ' ')} classes to {args.output_dir}."
    )
    return 0

class ExportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VisuoTactile Dataset Exporter")
        self.root.geometry("840x720")
        
        self.label_to_files = {}
        self.is_exporting = False
        
        self._create_widgets()
        
    def _create_widgets(self):
        # --- Directory Selection Frame ---
        dir_frame = ttk.LabelFrame(self.root, text="Directories", padding=10)
        dir_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Input Dir
        ttk.Label(dir_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_var = tk.StringVar(value=os.path.abspath('dataset'))
        ttk.Entry(dir_frame, textvariable=self.input_var, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(dir_frame, text="Browse", command=self.browse_input).grid(row=0, column=2, pady=2)
        
        # Output Dir
        ttk.Label(dir_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.output_var = tk.StringVar(value=os.path.abspath('ml_dataset'))
        ttk.Entry(dir_frame, textvariable=self.output_var, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(dir_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=2)
        
        # Scan Options
        scan_frame = ttk.Frame(dir_frame)
        scan_frame.grid(row=2, column=1, sticky=tk.W, pady=5)
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scan_frame, text="Scan Subfolders Recursively", variable=self.recursive_var).pack(side=tk.LEFT)
        
        self.include_video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scan_frame, text="Include Video Frames", variable=self.include_video_var).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(dir_frame, text="Scan Directory", command=self.scan_directory).grid(row=2, column=2, pady=5)
        
        # --- Settings Frame ---
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Split Ratios
        split_frame = ttk.Frame(settings_frame)
        split_frame.pack(fill=tk.X, pady=5)
        ttk.Label(split_frame, text="Splits (Train/Val/Test):").pack(side=tk.LEFT)
        self.train_var = tk.DoubleVar(value=0.8)
        self.val_var = tk.DoubleVar(value=0.1)
        self.test_var = tk.DoubleVar(value=0.1)
        ttk.Entry(split_frame, textvariable=self.train_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Entry(split_frame, textvariable=self.val_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Entry(split_frame, textvariable=self.test_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Export Mode
        mode_frame = ttk.LabelFrame(settings_frame, text="Export Mode", padding=5)
        mode_frame.pack(fill=tk.X, pady=5)
        self.category_var = tk.StringVar(value="local_feature")
        ttk.Radiobutton(
            mode_frame,
            text="Per Local Feature",
            variable=self.category_var,
            value="local_feature",
            command=self.scan_directory,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame,
            text="Per Object Label",
            variable=self.category_var,
            value="label",
            command=self.scan_directory,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            mode_frame,
            text="Output folders are named from the selected metadata field.",
        ).pack(side=tk.LEFT, padx=12)
        
        # Feature Suffix
        feat_frame = ttk.Frame(settings_frame)
        feat_frame.pack(fill=tk.X, pady=5)
        ttk.Label(feat_frame, text="Feature Suffix (e.g. raw.png):").pack(side=tk.LEFT)
        self.feature_var = tk.StringVar(value="raw.png")
        ttk.Entry(feat_frame, textvariable=self.feature_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Augmentations
        aug_frame = ttk.LabelFrame(settings_frame, text="Augmentations (applies to Train split only)", padding=5)
        aug_frame.pack(fill=tk.X, pady=5)
        
        self.aug_rot180_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(aug_frame, text="Rotate 180°", variable=self.aug_rot180_var).pack(side=tk.LEFT, padx=5)
        
        self.aug_hflip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(aug_frame, text="Horizontal Flip", variable=self.aug_hflip_var).pack(side=tk.LEFT, padx=5)
        
        self.aug_vflip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(aug_frame, text="Vertical Flip", variable=self.aug_vflip_var).pack(side=tk.LEFT, padx=5)
        
        # --- Export & Progress ---
        export_frame = ttk.Frame(self.root, padding=10)
        export_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.export_btn = ttk.Button(export_frame, text="Start Export", command=self.start_export_thread)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(export_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(export_frame, textvariable=self.status_var).pack(side=tk.RIGHT, padx=5)

        # --- Info Frame (Classes & Preview) ---
        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Classes Listbox
        self.class_frame = ttk.LabelFrame(info_frame, text="Detected Local Features", padding=5)
        class_frame = self.class_frame
        class_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.class_listbox = tk.Listbox(class_frame)
        self.class_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(class_frame, orient=tk.VERTICAL, command=self.class_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.class_listbox.config(yscrollcommand=scrollbar.set)
        
        # Preview Structure
        self.preview_frame = ttk.LabelFrame(info_frame, text="Local-Feature Output Preview", padding=5)
        preview_frame = self.preview_frame
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.preview_text = tk.Text(preview_frame, width=40, state=tk.DISABLED)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        self.refresh_export_mode_labels()

    def browse_input(self):
        dir_path = filedialog.askdirectory(initialdir=self.input_var.get())
        if dir_path:
            self.input_var.set(dir_path)

    def browse_output(self):
        dir_path = filedialog.askdirectory(initialdir=self.output_var.get())
        if dir_path:
            self.output_var.set(dir_path)

    def selected_export_label(self):
        if self.category_var.get() == "local_feature":
            return "local features"
        return "object labels"

    def refresh_export_mode_labels(self):
        is_local_feature = self.category_var.get() == "local_feature"
        export_label = self.selected_export_label().title()
        self.class_frame.config(text=f"Detected {export_label}")
        self.preview_frame.config(text=f"{export_label} Output Preview")
        if hasattr(self, "export_btn"):
            button_text = (
                "Export Local-Feature Dataset"
                if is_local_feature
                else "Export Object-Label Dataset"
            )
            self.export_btn.config(text=button_text)

    def scan_directory(self):
        self.status_var.set("Scanning directory...")
        self.root.update_idletasks()
        
        input_path = Path(self.input_var.get())
        if not input_path.exists():
            messagebox.showerror("Error", f"Input directory {input_path} does not exist.")
            self.status_var.set("Error scanning.")
            return

        feature_suffix = self.feature_var.get()
        category_by = self.category_var.get()
        self.label_to_files = scan_dataset(
            input_path,
            category_by=category_by,
            feature_suffix=feature_suffix,
            recursive=self.recursive_var.get(),
            include_video=self.include_video_var.get(),
        )

        self.refresh_export_mode_labels()
        self.update_class_list()
        self.update_preview()
        
        total_files = sum(len(files) for files in self.label_to_files.values())
        self.status_var.set(
            f"Found {total_files} valid files across "
            f"{len(self.label_to_files)} {self.selected_export_label()}."
        )
        
        # Export labels to labels.txt in the input directory immediately after scanning
        try:
            labels_path = input_path / "labels.txt"
            with open(labels_path, "w") as f:
                for label in sorted(self.label_to_files.keys()):
                    f.write(f"{label}\n")
        except Exception as e:
            print(f"Failed to write labels.txt to input directory: {e}")

    def update_class_list(self):
        self.class_listbox.delete(0, tk.END)
        for label, files in sorted(self.label_to_files.items()):
            group_count = len({record.group_id for record in files})
            self.class_listbox.insert(
                tk.END,
                f"{label} ({len(files)} files, {group_count} acquisition groups)",
            )

    def update_preview(self):
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        
        if not self.label_to_files:
            self.refresh_export_mode_labels()
            self.preview_text.insert(
                tk.END,
                f"No {self.selected_export_label()} found to export.",
            )
            self.preview_text.config(state=tk.DISABLED)
            return

        self.refresh_export_mode_labels()
        preview = f"{Path(self.output_var.get()).name}/\n"
        for split in ['train', 'val', 'test']:
            preview += f"├── {split}/\n"
            for i, label in enumerate(sorted(self.label_to_files.keys())[:3]): # Show top 3
                preview += f"│   ├── {label}/\n"
            if len(self.label_to_files) > 3:
                preview += f"│   ├── ...\n"
        
        self.preview_text.insert(tk.END, preview)
        self.preview_text.config(state=tk.DISABLED)

    def start_export_thread(self):
        if self.is_exporting:
            return
            
        train_split = self.train_var.get()
        val_split = self.val_var.get()
        test_split = self.test_var.get()
        
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            messagebox.showerror("Error", "Train/Val/Test split ratios must sum to 1.0")
            return
            
        if not self.label_to_files:
            messagebox.showwarning("Warning", "No files to export. Please scan the directory first.")
            return

        try:
            validate_output_directory(self.output_var.get())
        except ValueError as exc:
            messagebox.showerror("Export destination", str(exc))
            return

        self.is_exporting = True
        self.export_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        threading.Thread(target=self.export_dataset, daemon=True).start()

    def process_image(self, src_file, dst_file, augmentation=None):
        try:
            process_image_file(src_file, dst_file, augmentation)
        except Exception as e:
            print(f"Failed to process {src_file}: {e}")

    def export_dataset(self):
        output_path = Path(self.output_var.get())
        
        for split_dir in ['train', 'val', 'test']:
            os.makedirs(output_path / split_dir, exist_ok=True)
            
        total_files_to_copy = sum(len(files) for files in self.label_to_files.values())
        if total_files_to_copy == 0:
            self.finish_export("No files to export.")
            return
            
        # Calculate max possible operations to scale progress
        operations_per_file = 1
        if self.aug_rot180_var.get(): operations_per_file += 1
        if self.aug_hflip_var.get(): operations_per_file += 1
        if self.aug_vflip_var.get(): operations_per_file += 1
        total_operations = total_files_to_copy * operations_per_file
        
        processed_ops = 0
        split_manifest = {
            "schema_version": 1,
            "category_by": self.category_var.get(),
            "feature_suffix": self.feature_var.get(),
            "grouping": "acquisition",
            "seed": 42,
            "ratios": {
                "train": self.train_var.get(),
                "val": self.val_var.get(),
                "test": self.test_var.get(),
            },
            "classes": {},
        }

        with ThreadPoolExecutor() as executor:
            futures = []
            for label, files_list in self.label_to_files.items():
                splits = grouped_split(
                    files_list,
                    train_ratio=self.train_var.get(),
                    val_ratio=self.val_var.get(),
                    test_ratio=self.test_var.get(),
                    seed=42,
                )
                split_manifest["classes"][label] = {
                    split_name: sorted({record.group_id for record in subset})
                    for split_name, subset in splits.items()
                }
                
                for split_name, subset in splits.items():
                    if not subset:
                        continue
                        
                    target_dir = output_path / split_name / label
                    os.makedirs(target_dir, exist_ok=True)
                    
                    for record in subset:
                        src_file = record.source_file
                        sensor_prefix = record.sensor_prefix
                        # Base File
                        base_name = f"{sensor_prefix}_{src_file.name}"
                        dst_file = target_dir / base_name
                        futures.append(executor.submit(self.process_image, src_file, dst_file))
                        
                        # Augmentations (Only apply to Training set to prevent data leakage/test contamination)
                        if split_name == 'train':
                            name_no_ext, ext = os.path.splitext(src_file.name)
                            
                            if self.aug_rot180_var.get():
                                aug_file = target_dir / f"{sensor_prefix}_{name_no_ext}_rot180{ext}"
                                futures.append(executor.submit(self.process_image, src_file, aug_file, 'rot180'))
                                
                            if self.aug_hflip_var.get():
                                aug_file = target_dir / f"{sensor_prefix}_{name_no_ext}_hflip{ext}"
                                futures.append(executor.submit(self.process_image, src_file, aug_file, 'hflip'))
                                
                            if self.aug_vflip_var.get():
                                aug_file = target_dir / f"{sensor_prefix}_{name_no_ext}_vflip{ext}"
                                futures.append(executor.submit(self.process_image, src_file, aug_file, 'vflip'))
                        else:
                            # Fast forward progress for val/test that skip augmentations
                            processed_ops += (operations_per_file - 1)
                            
            for future in as_completed(futures):
                future.result()
                processed_ops += 1
                # Update UI progress
                progress_pct = (processed_ops / total_operations) * 100
                self.root.after(0, self.progress_var.set, progress_pct)
                self.root.after(0, self.status_var.set, f"Exporting... {progress_pct:.1f}%")

        labels = sorted(list(self.label_to_files.keys()))
        labels_file = output_path / "labels.txt"
        try:
            with open(labels_file, "w") as f:
                for label in labels:
                    f.write(f"{label}\n")
        except Exception as e:
            print(f"Failed to write labels.txt: {e}")

        manifest_path = output_path / "split_manifest.json"
        try:
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(split_manifest, handle, indent=4)
                handle.write("\n")
        except Exception as e:
            print(f"Failed to write split manifest: {e}")
            
        self.root.after(0, self.finish_export, "Export Completed Successfully!")

    def finish_export(self, message):
        self.is_exporting = False
        self.export_btn.config(state=tk.NORMAL)
        self.status_var.set(message)
        if "Completed" in message:
            messagebox.showinfo("Export Complete", message)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(run_cli(sys.argv[1:]))

    # Enable High DPI awareness on Windows
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()

    # Enable High DPI awareness on Linux
    if sys.platform.startswith("linux"):
        try:
            import subprocess
            xdb = subprocess.check_output(["xrdb", "-query"]).decode()
            for line in xdb.splitlines():
                if "Xft.dpi:" in line:
                    dpi = float(line.split(":")[1].strip())
                    root.tk.call("tk", "scaling", dpi / 72.0)
                    break
        except Exception:
            pass

    app = ExportApp(root)
    root.mainloop()
