import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, ImageTk
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

class TestUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ResNet Tester")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f8f9fa")

        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#f8f9fa", foreground="#212529", font=("Segoe UI", 10))
        self.style.configure("TLabel", background="#f8f9fa", font=("Segoe UI", 10))
        self.style.configure("TButton", background="#007bff", foreground="#ffffff", font=("Segoe UI", 10, "bold"), padding=5)
        self.style.map("TButton", background=[("active", "#0056b3")])
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        self.style.configure("TFrame", background="#f8f9fa")
        self.style.configure("TLabelframe", background="#f8f9fa")
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"), background="#f8f9fa")

        # Variables
        self.model_path = tk.StringVar()
        self.labels_path = tk.StringVar()
        self.dataset_path = tk.StringVar()
        self.model_arch = tk.StringVar(value="resnet18")
        self.class_names = []
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.current_idx = 0
        self.filtered_indices = []
        self.predictions_cache = {}  # Store predictions so we don't recompute per image when stepping

        self.setup_ui()
        
    def setup_ui(self):
        # Main Layout
        self.sidebar = ttk.Frame(self.root, padding=10)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        self.main_area = ttk.Frame(self.root, padding=10)
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Split main area into Top (Viewer) and Bottom (Metrics)
        self.top_frame = ttk.Frame(self.main_area)
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.bottom_frame = ttk.Frame(self.main_area)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.setup_sidebar()
        self.setup_image_viewer()
        self.setup_metrics_area()

    def setup_sidebar(self):
        ttk.Label(self.sidebar, text="Configuration", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 15))

        # Model Arch
        ttk.Label(self.sidebar, text="Model Architecture:").pack(anchor=tk.W)
        arch_cb = ttk.Combobox(self.sidebar, textvariable=self.model_arch, state="readonly", 
                               values=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
        arch_cb.pack(fill=tk.X, pady=(0, 15))

        # Browse Model
        ttk.Label(self.sidebar, text="Trained Weights (.pth):").pack(anchor=tk.W)
        model_frame = ttk.Frame(self.sidebar)
        model_frame.pack(fill=tk.X, pady=(0, 5))
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path, state='readonly')
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="Browse", width=8, command=self.browse_model).pack(side=tk.RIGHT, padx=(5, 0))

        # Labels found status
        self.labels_status = ttk.Label(self.sidebar, text="Labels: Not loaded", foreground="red")
        self.labels_status.pack(anchor=tk.W, pady=(0, 15))

        # Browse Dataset
        ttk.Label(self.sidebar, text="Dataset Folder (e.g. val):").pack(anchor=tk.W)
        dataset_frame = ttk.Frame(self.sidebar)
        dataset_frame.pack(fill=tk.X, pady=(0, 15))
        self.dataset_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_path, state='readonly')
        self.dataset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dataset_frame, text="Browse", width=8, command=self.browse_dataset).pack(side=tk.RIGHT, padx=(5, 0))

        # Run Validation Button
        self.run_btn = ttk.Button(self.sidebar, text="Run Validation on Dataset", command=self.run_validation_thread)
        self.run_btn.pack(fill=tk.X, pady=(20, 5))

        # Progress bar
        self.progress = ttk.Progressbar(self.sidebar, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(5, 10))
        self.status_label = ttk.Label(self.sidebar, text="Ready")
        self.status_label.pack(anchor=tk.W)

    def setup_image_viewer(self):
        self.img_frame = ttk.LabelFrame(self.top_frame, text="Image Inspector", padding=10)
        self.img_frame.pack(fill=tk.BOTH, expand=True)

        self.img_label = ttk.Label(self.img_frame, text="No Image Loaded")
        self.img_label.pack(side=tk.LEFT, expand=True)
        
        # middle: browser list
        self.browser_frame = ttk.Frame(self.img_frame, padding=(10, 0))
        self.browser_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(self.browser_frame, text="Filter by Class:").pack(anchor=tk.W)
        self.category_var = tk.StringVar(value="All")
        self.category_cb = ttk.Combobox(self.browser_frame, textvariable=self.category_var, state="readonly")
        self.category_cb.pack(fill=tk.X, pady=(0, 10))
        self.category_cb.bind("<<ComboboxSelected>>", self.on_category_select)
        
        list_frame = ttk.Frame(self.browser_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.photo_list = tk.Listbox(list_frame, width=30)
        self.photo_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.photo_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.photo_list.config(yscrollcommand=scrollbar.set)
        self.photo_list.bind("<<ListboxSelect>>", self.on_listbox_select)

        self.info_frame = ttk.Frame(self.img_frame, padding=20)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.true_label_var = tk.StringVar(value="True Label: N/A")
        self.pred_label_var = tk.StringVar(value="Predicted: N/A")
        self.conf_var = tk.StringVar(value="Confidence: N/A")
        self.file_var = tk.StringVar(value="File: N/A")

        ttk.Label(self.info_frame, textvariable=self.file_var, font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(self.info_frame, textvariable=self.true_label_var, font=("Segoe UI", 12, "bold"), foreground="#007bff").pack(anchor=tk.W, pady=(0, 5))
        self.pred_ui_label = ttk.Label(self.info_frame, textvariable=self.pred_label_var, font=("Segoe UI", 12, "bold"))
        self.pred_ui_label.pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(self.info_frame, textvariable=self.conf_var).pack(anchor=tk.W, pady=(0, 20))

        nav_frame = ttk.Frame(self.info_frame)
        nav_frame.pack(anchor=tk.W)
        ttk.Button(nav_frame, text="<< Prev", command=self.prev_image, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next >>", command=self.next_image, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Random", command=self.random_image, width=8).pack(side=tk.LEFT)

    def setup_metrics_area(self):
        metrics_lf = ttk.LabelFrame(self.bottom_frame, text="Metrics", padding=10)
        metrics_lf.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.text_frame = ttk.Frame(metrics_lf)
        self.text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.metrics_text = tk.Text(self.text_frame, height=20, font=("Consolas", 9))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.insert(tk.END, "Run validation to see metrics here...")
        self.metrics_text.config(state=tk.DISABLED)

    def browse_model(self):
        path = filedialog.askopenfilename(title="Select Model Weights", filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")])
        if path:
            self.model_path.set(path)
            self.model = None # Reset model
            
            # Auto-detect ResNet architecture
            try:
                state_dict = torch.load(path, map_location="cpu")
                keys = state_dict.keys()
                has_bottleneck = any("layer1.0.conv3.weight" in k for k in keys)
                
                max_block = -1
                for k in keys:
                    if k.startswith("layer3."):
                        parts = k.split(".")
                        if len(parts) > 1 and parts[1].isdigit():
                            max_block = max(max_block, int(parts[1]))
                
                detected_arch = None
                if has_bottleneck:
                    if max_block >= 35: detected_arch = "resnet152"
                    elif max_block >= 22: detected_arch = "resnet101"
                    else: detected_arch = "resnet50"
                else:
                    if max_block >= 5: detected_arch = "resnet34"
                    elif max_block >= 1: detected_arch = "resnet18"
                        
                if detected_arch:
                    self.model_arch.set(detected_arch)
            except Exception:
                pass # Silent fallback if detection fails
                
            # Look for labels.txt
            expected_labels = os.path.splitext(path)[0] + ".txt"
            if os.path.exists(expected_labels):
                self.labels_path.set(expected_labels)
            else:
                dir_path = os.path.dirname(path)
                fallback = os.path.join(dir_path, "labels.txt")
                if os.path.exists(fallback):
                    self.labels_path.set(fallback)
                else:
                    self.labels_path.set("")
            
            if self.labels_path.get():
                try:
                    with open(self.labels_path.get(), "r") as f:
                        self.class_names = [line.strip() for line in f if line.strip()]
                    self.labels_status.config(text=f"Labels: {len(self.class_names)} classes loaded", foreground="green")
                except Exception as e:
                    self.labels_status.config(text=f"Error loading labels", foreground="red")
            else:
                self.labels_status.config(text="Labels: Not found!", foreground="red")

    def browse_dataset(self):
        path = filedialog.askdirectory(title="Select Dataset Split Directory")
        if path:
            self.dataset_path.set(path)
            self.load_dataset()

    def load_dataset(self):
        path = self.dataset_path.get()
        if not path or not os.path.isdir(path):
            return
            
        # Check if user selected the parent dataset directory instead of a split
        try:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            # If the directory contains split folders but not class folders
            if set(subdirs).intersection({'train', 'val', 'test'}) and (not self.class_names or not set(self.class_names).intersection(subdirs)):
                preferred_split = 'test' if 'test' in subdirs else ('val' if 'val' in subdirs else 'train')
                path = os.path.join(path, preferred_split)
                self.dataset_path.set(path)
                messagebox.showinfo("Auto-corrected Dataset Path", f"You selected a directory containing dataset splits. Automatically selecting the '{preferred_split}' split:\n{path}")
        except Exception:
            pass

        data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        try:
            self.dataset = datasets.ImageFolder(path, data_transform)
            if len(self.dataset) == 0:
                messagebox.showwarning("Warning", "Dataset is empty.")
                return
            
            # verify class names match model
            if self.class_names:
                ds_classes = self.dataset.classes
                if ds_classes != self.class_names:
                    messagebox.showwarning("Warning", f"Dataset classes ({len(ds_classes)}) do not match model labels ({len(self.class_names)})!\n\nDataset: {ds_classes}\nModel: {self.class_names}")

            self.category_cb['values'] = ["All"] + self.dataset.classes
            self.category_var.set("All")
            self.predictions_cache.clear()
            self.on_category_select()
            
            self.status_label.config(text=f"Dataset loaded: {len(self.dataset)} images")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def load_model(self):
        if not self.model_path.get() or not self.class_names:
            raise ValueError("Model path and labels must be loaded first.")
            
        if self.model is None:
            self.status_label.config(text="Loading model architecture...")
            self.root.update_idletasks()
            arch = self.model_arch.get()
            if not hasattr(models, arch):
                raise ValueError(f"Unknown architecture: {arch}")
            
            model_fn = getattr(models, arch)
            model = model_fn(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))
            
            self.status_label.config(text="Loading weights...")
            self.root.update_idletasks()
            model.load_state_dict(torch.load(self.model_path.get(), map_location=self.device))
            model = model.to(self.device)
            model.eval()
            self.model = model
            self.status_label.config(text="Model loaded successfully.")

    def run_validation_thread(self):
        if not self.model_path.get() or not self.class_names:
            messagebox.showwarning("Missing Configuration", "Please load a model and dataset first.")
            return
        if not self.dataset:
            messagebox.showwarning("Missing Configuration", "Please load a dataset first.")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.progress['maximum'] = len(self.dataset)
        
        thread = threading.Thread(target=self._validation_worker, daemon=True)
        thread.start()

    def _validation_worker(self):
        try:
            self.load_model()
            
            dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False, num_workers=0)
            
            all_preds = []
            all_labels = []
            
            self.status_label.config(text="Validating...")
            
            processed = 0
            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    
                    processed += inputs.size(0)
                    # Update progress in main thread safely
                    self.root.after(0, self._update_progress, processed)
            
            self.root.after(0, self._display_metrics, all_labels, all_preds)
        
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Validation failed: {e}")
        finally:
            self.root.after(0, self._reset_run_btn)

    def _update_progress(self, val):
        self.progress['value'] = val

    def _reset_run_btn(self):
        self.run_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Validation Complete")

    def _display_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.dataset.classes, zero_division=0)
        
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, f"Overall Accuracy: {acc:.4f}\n\n")
        self.metrics_text.insert(tk.END, report)
        self.metrics_text.config(state=tk.DISABLED)

    def update_image_view(self):
        if not self.dataset:
            return
            
        img_path, label_idx = self.dataset.samples[self.current_idx]
        true_label = self.dataset.classes[label_idx]
        
        # Display image
        try:
            pil_img = Image.open(img_path).convert('RGB')
            # Resize for display
            pil_img = pil_img.resize((300, 300), Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.img_label.config(image=self.tk_img, text="")
        except Exception as e:
            self.img_label.config(image='', text=f"Error loading image:\n{e}")

        filename = os.path.basename(img_path)
        self.file_var.set(f"File: {filename} ({self.current_idx + 1}/{len(self.dataset)})")
        self.true_label_var.set(f"True Label: {true_label}")
        
        # If we have a model, run a quick inference
        if self.model_path.get() and self.class_names:
            if self.current_idx in self.predictions_cache:
                pred_label, conf = self.predictions_cache[self.current_idx]
                self._update_pred_ui(pred_label, true_label, conf)
            else:
                self.pred_label_var.set("Predicted: Computing...")
                self.conf_var.set("Confidence: ...")
                self.pred_ui_label.config(foreground="black")
                self.root.update_idletasks()
                
                # run inference in background or quickly here since it's batch size 1
                try:
                    if self.model is None:
                        self.load_model()
                    
                    img_tensor, _ = self.dataset[self.current_idx]
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        out = self.model(img_tensor)
                        probs = torch.nn.functional.softmax(out, dim=1)
                        conf, pred = torch.max(probs, 1)
                        
                        pred_idx = pred.item()
                        conf_val = conf.item()
                        
                        # Fallback if model classes != dataset classes
                        if pred_idx < len(self.class_names):
                            pred_label = self.class_names[pred_idx]
                        else:
                            pred_label = f"Unknown ({pred_idx})"
                            
                        self.predictions_cache[self.current_idx] = (pred_label, conf_val)
                        self._update_pred_ui(pred_label, true_label, conf_val)
                except Exception as e:
                    self.pred_label_var.set("Predicted: Error")
                    self.conf_var.set(str(e))
        else:
            self.pred_label_var.set("Predicted: N/A (Load Model)")
            self.conf_var.set("Confidence: N/A")
            self.pred_ui_label.config(foreground="black")

    def _update_pred_ui(self, pred_label, true_label, conf):
        self.pred_label_var.set(f"Predicted: {pred_label}")
        self.conf_var.set(f"Confidence: {conf:.2%}")
        
        if pred_label == true_label:
            self.pred_ui_label.config(foreground="green")
        else:
            self.pred_ui_label.config(foreground="red")

    def on_category_select(self, event=None):
        if not self.dataset: return
        cat = self.category_var.get()
        self.photo_list.delete(0, tk.END)
        self.filtered_indices = []
        for i, (path, label_idx) in enumerate(self.dataset.samples):
            cls_name = self.dataset.classes[label_idx]
            if cat == "All" or cat == cls_name:
                self.filtered_indices.append(i)
                self.photo_list.insert(tk.END, os.path.basename(path))
        
        if self.filtered_indices:
            self.photo_list.selection_set(0)
            self.current_idx = self.filtered_indices[0]
            self.update_image_view()

    def on_listbox_select(self, event):
        sel = self.photo_list.curselection()
        if not sel: return
        list_idx = sel[0]
        self.current_idx = self.filtered_indices[list_idx]
        self.update_image_view()

    def random_image(self):
        if self.filtered_indices:
            import random
            list_idx = random.randint(0, len(self.filtered_indices)-1)
            self.photo_list.selection_clear(0, tk.END)
            self.photo_list.selection_set(list_idx)
            self.photo_list.see(list_idx)
            self.current_idx = self.filtered_indices[list_idx]
            self.update_image_view()

    def prev_image(self):
        if not self.filtered_indices: return
        try:
            curr_list_idx = self.filtered_indices.index(self.current_idx)
        except ValueError:
            curr_list_idx = 0
            
        if curr_list_idx > 0:
            new_idx = curr_list_idx - 1
            self.photo_list.selection_clear(0, tk.END)
            self.photo_list.selection_set(new_idx)
            self.photo_list.see(new_idx)
            self.current_idx = self.filtered_indices[new_idx]
            self.update_image_view()

    def next_image(self):
        if not self.filtered_indices: return
        try:
            curr_list_idx = self.filtered_indices.index(self.current_idx)
        except ValueError:
            curr_list_idx = 0
            
        if curr_list_idx < len(self.filtered_indices) - 1:
            new_idx = curr_list_idx + 1
            self.photo_list.selection_clear(0, tk.END)
            self.photo_list.selection_set(new_idx)
            self.photo_list.see(new_idx)
            self.current_idx = self.filtered_indices[new_idx]
            self.update_image_view()

if __name__ == "__main__":
    # Enable High DPI awareness on Windows
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2) # Per Monitor V2
    except Exception:
        pass

    root = tk.Tk()
    
    # Enable High DPI awareness on Linux
    import sys
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
            
    app = TestUIApp(root)
    root.mainloop()
