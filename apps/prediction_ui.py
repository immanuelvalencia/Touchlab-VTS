import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import importlib.util
import pathlib
import sys
from PIL import Image, ImageTk

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visuo-Tactile Model Prediction")
        self.root.geometry("800x650")
        self.root.configure(bg="#f8f9fa")

        # White/Light UI Styles matching Dataset Labeling App
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#f8f9fa", foreground="#212529", fieldbackground="#ffffff")
        self.style.configure("TLabel", background="#f8f9fa", foreground="#212529", font=("Segoe UI", 11))
        self.style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), background="#f8f9fa")
        self.style.configure("TButton", background="#007bff", foreground="#ffffff", borderwidth=0, font=("Segoe UI", 10, "bold"), padding=6)
        self.style.map("TButton", background=[("active", "#0056b3")])
        self.style.configure("TCombobox", fieldbackground="#ffffff", background="#e9ecef", foreground="#212529", font=("Segoe UI", 11))
        self.style.configure("TFrame", background="#f8f9fa")
        self.style.configure("Card.TFrame", background="#ffffff", borderwidth=1, relief="solid")

        # Variables
        self.model_var = tk.StringVar()
        self.img_path_var = tk.StringVar()
        self.result_var = tk.StringVar(value="Waiting for input...")
        self.image_tk = None

        self.create_widgets()

    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        ttk.Label(header_frame, text="Visuo-Tactile Inference", style="Title.TLabel").pack(side=tk.LEFT)

        # Main Container (Card style)
        main_frame = tk.Frame(self.root, bg="#ffffff", highlightbackground="#dee2e6", highlightthickness=1)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # --- Controls Area ---
        controls_frame = tk.Frame(main_frame, bg="#ffffff")
        controls_frame.pack(fill=tk.X, padx=20, pady=20)

        # Model selector
        ttk.Label(controls_frame, text="Select Model Script:", background="#ffffff").grid(row=0, column=0, sticky="w", pady=10)
        
        model_path_frame = tk.Frame(controls_frame, bg="#ffffff")
        model_path_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=10)
        
        ttk.Entry(model_path_frame, textvariable=self.model_var, width=50, font=("Segoe UI", 10)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="Browse...", command=self.browse_model).pack(side=tk.LEFT, padx=(10, 0))

        # Image selector
        ttk.Label(controls_frame, text="Image to predict:", background="#ffffff").grid(row=1, column=0, sticky="w", pady=10)
        
        path_frame = tk.Frame(controls_frame, bg="#ffffff")
        path_frame.grid(row=1, column=1, sticky="ew", padx=10, pady=10)
        
        ttk.Entry(path_frame, textvariable=self.img_path_var, width=50, font=("Segoe UI", 10)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="Browse...", command=self.browse_image).pack(side=tk.LEFT, padx=(10, 0))

        # Predict button
        ttk.Button(controls_frame, text="Run Prediction", command=self.on_predict).grid(row=2, column=0, columnspan=2, pady=20)

        # --- Image Preview Area ---
        preview_frame = tk.Frame(main_frame, bg="#f8f9fa", highlightbackground="#e9ecef", highlightthickness=1)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.image_label = tk.Label(preview_frame, text="No Image Selected", bg="#f8f9fa", fg="#6c757d", font=("Segoe UI", 12, "italic"))
        self.image_label.pack(expand=True)

        # --- Result Area ---
        result_frame = tk.Frame(self.root, bg="#007bff")
        result_frame.pack(fill=tk.X)
        
        ttk.Label(result_frame, textvariable=self.result_var, font=("Segoe UI", 14, "bold"), 
                  background="#007bff", foreground="#ffffff", padding=15).pack()

    def run_model_script(self, file_path: str, image_path: str):
        try:
            spec = importlib.util.spec_from_file_location("dynamic_model", file_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "predict"):
                raise AttributeError(f"{file_path} has no `predict` function.")
            return mod.predict(image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run {file_path}:\n{e}")
            return None

    def browse_model(self):
        initial_dir = str(pathlib.Path(__file__).parent.parent)
        filename = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select a Model Script",
            filetypes=[("Python files", "*.py")],
        )
        if filename:
            self.model_var.set(filename)
            self.result_var.set("Model script selected")

    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if filename:
            self.img_path_var.set(filename)
            self.show_preview(filename)
            self.result_var.set("Ready to predict")

    def show_preview(self, filename):
        try:
            img = Image.open(filename)
            # Resize for preview
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.image_tk, text="")
        except Exception as e:
            self.image_label.configure(image="", text="Could not load image preview")

    def on_predict(self):
        model_script = self.model_var.get()
        img_path = self.img_path_var.get()
        if not model_script:
            messagebox.showwarning("Select model", "Please choose a model script.")
            return
        if not img_path:
            messagebox.showwarning("Select image", "Please choose an image file.")
            return

        self.result_var.set("Predicting... Please wait.")
        self.root.update()

        result = self.run_model_script(model_script, img_path)
        if result is not None:
            self.result_var.set(result)
        else:
            self.result_var.set("Prediction failed.")

def build_ui():
    # Ensure the root of the project is in the python path
    project_root = str(pathlib.Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    build_ui()
