import os
import shutil
import random
import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image

class ExportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VisuoTactile Dataset Exporter")
        self.root.geometry("800x700")
        
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
        class_frame = ttk.LabelFrame(info_frame, text="Detected Classes", padding=5)
        class_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.class_listbox = tk.Listbox(class_frame)
        self.class_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(class_frame, orient=tk.VERTICAL, command=self.class_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.class_listbox.config(yscrollcommand=scrollbar.set)
        
        # Preview Structure
        preview_frame = ttk.LabelFrame(info_frame, text="Output Structure Preview", padding=5)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.preview_text = tk.Text(preview_frame, width=40, state=tk.DISABLED)
        self.preview_text.pack(fill=tk.BOTH, expand=True)

    def browse_input(self):
        dir_path = filedialog.askdirectory(initialdir=self.input_var.get())
        if dir_path:
            self.input_var.set(dir_path)

    def browse_output(self):
        dir_path = filedialog.askdirectory(initialdir=self.output_var.get())
        if dir_path:
            self.output_var.set(dir_path)

    def scan_directory(self):
        self.status_var.set("Scanning directory...")
        self.root.update_idletasks()
        
        input_path = Path(self.input_var.get())
        if not input_path.exists():
            messagebox.showerror("Error", f"Input directory {input_path} does not exist.")
            self.status_var.set("Error scanning.")
            return

        self.label_to_files = {}
        feature_suffix = self.feature_var.get()
        include_video = self.include_video_var.get()
        
        if self.recursive_var.get():
            metadata_files = list(input_path.rglob("*metadata.json"))
        else:
            metadata_files = list(input_path.glob("*metadata.json"))

        for meta_file in metadata_files:
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            if not include_video and meta.get("is_video_sequence", False):
                continue
                
            label = meta.get("label", "Unknown")
            saved_features = meta.get("saved_features", [])
            
            target_file = None
            for saved_f in saved_features:
                if saved_f.endswith(feature_suffix):
                    target_file = meta_file.parent / saved_f
                    break
                    
            if target_file and target_file.exists():
                if label not in self.label_to_files:
                    self.label_to_files[label] = []
                sensor_prefix = meta.get("sensor", "UnknownSensor")
                self.label_to_files[label].append((target_file, sensor_prefix))

        self.update_class_list()
        self.update_preview()
        
        total_files = sum(len(files) for files in self.label_to_files.values())
        self.status_var.set(f"Found {total_files} valid files across {len(self.label_to_files)} classes.")

    def update_class_list(self):
        self.class_listbox.delete(0, tk.END)
        for label, files in sorted(self.label_to_files.items()):
            self.class_listbox.insert(tk.END, f"{label} ({len(files)} files)")

    def update_preview(self):
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        
        if not self.label_to_files:
            self.preview_text.insert(tk.END, "No classes found to export.")
            self.preview_text.config(state=tk.DISABLED)
            return

        preview = "ml_dataset/\n"
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

        self.is_exporting = True
        self.export_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        threading.Thread(target=self.export_dataset, daemon=True).start()

    def process_image(self, src_file, dst_file, augmentation=None):
        try:
            if augmentation:
                with Image.open(src_file) as img:
                    if augmentation == 'rot180':
                        img = img.rotate(180)
                    elif augmentation == 'hflip':
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif augmentation == 'vflip':
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    img.save(dst_file)
            else:
                shutil.copy2(src_file, dst_file)
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

        for label, files_list in self.label_to_files.items():
            random.seed(42) # Consistent splitting
            random.shuffle(files_list)
            
            total_files = len(files_list)
            train_idx = int(total_files * self.train_var.get())
            val_idx = train_idx + int(total_files * self.val_var.get())
            
            splits = {
                'train': files_list[:train_idx],
                'val': files_list[train_idx:val_idx],
                'test': files_list[val_idx:]
            }
            
            for split_name, subset in splits.items():
                if not subset:
                    continue
                    
                target_dir = output_path / split_name / label
                os.makedirs(target_dir, exist_ok=True)
                
                for src_file, sensor_prefix in subset:
                    # Base File
                    base_name = f"{sensor_prefix}_{src_file.name}"
                    dst_file = target_dir / base_name
                    self.process_image(src_file, dst_file)
                    
                    processed_ops += 1
                    
                    # Augmentations (Only apply to Training set to prevent data leakage/test contamination)
                    if split_name == 'train':
                        name_no_ext, ext = os.path.splitext(src_file.name)
                        
                        if self.aug_rot180_var.get():
                            aug_file = target_dir / f"{sensor_prefix}_{name_no_ext}_rot180{ext}"
                            self.process_image(src_file, aug_file, 'rot180')
                            processed_ops += 1
                            
                        if self.aug_hflip_var.get():
                            aug_file = target_dir / f"{sensor_prefix}_{name_no_ext}_hflip{ext}"
                            self.process_image(src_file, aug_file, 'hflip')
                            processed_ops += 1
                            
                        if self.aug_vflip_var.get():
                            aug_file = target_dir / f"{sensor_prefix}_{name_no_ext}_vflip{ext}"
                            self.process_image(src_file, aug_file, 'vflip')
                            processed_ops += 1
                    else:
                        # Fast forward progress for val/test that skip augmentations
                        processed_ops += (operations_per_file - 1)

                    # Update UI progress
                    progress_pct = (processed_ops / total_operations) * 100
                    self.root.after(0, self.progress_var.set, progress_pct)
                    self.root.after(0, self.status_var.set, f"Exporting... {progress_pct:.1f}%")

        self.root.after(0, self.finish_export, "Export Completed Successfully!")

    def finish_export(self, message):
        self.is_exporting = False
        self.export_btn.config(state=tk.NORMAL)
        self.status_var.set(message)
        if "Completed" in message:
            messagebox.showinfo("Export Complete", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ExportApp(root)
    root.mainloop()
