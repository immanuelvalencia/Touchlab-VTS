import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from scipy.fft import dst, idst
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import threading
import time
import os
import json
import importlib
import sys
import pathlib
import os
import sys
import pathlib

# Add the root 'visuotactile' directory to sys.path so 'apps.models' can be found
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


MODEL_MODULES = {
    "ResNet-18": "apps.models.resnet18_predict",
}
import importlib
import sys
import pathlib
import os
import sys
import pathlib

# Add the root 'visuotactile' directory to sys.path so 'apps.models' can be found
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


MODEL_MODULES = {
    "ResNet-18": "apps.models.resnet18_predict",
}

# --- Fast Poisson Solver using 2D DST-I ---
def solve_poisson_dst(gx, gy):
    """
    Solves 2D Poisson equation: Laplacian(Z) = div(gx, gy)
    using 2D Discrete Sine Transform (DST-I) for Dirichlet boundary conditions.
    """
    m, n = gx.shape
    # Compute divergence of gradients
    f = np.zeros((m, n))
    f[:, 1:] += gx[:, 1:] - gx[:, :-1]
    f[:, 0] += gx[:, 0]
    f[1:, :] += gy[1:, :] - gy[:-1, :]
    f[0, :] += gy[0, :]
    
    # 2D DST-I
    f_dst = dst(dst(f, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
    
    # Eigenvalues of 2D Laplacian for Dirichlet boundary conditions
    y = np.arange(1, m + 1).reshape(-1, 1)
    x = np.arange(1, n + 1).reshape(1, -1)
    denom = 2 * np.cos(np.pi * y / (m + 1)) + 2 * np.cos(np.pi * x / (n + 1)) - 4
    
    # Avoid division by zero
    denom[denom == 0] = 1.0
    
    u_dst = f_dst / denom
    
    # 2D IDST-I
    u = idst(idst(u_dst, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
    return u

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visuo-Tactile Prediction App")
        self.root.geometry("1300x840")
        self.root.configure(bg="#f8f9fa")
        
        # White/Light UI Styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#f8f9fa", foreground="#212529", fieldbackground="#ffffff")
        self.style.configure("TLabel", background="#f8f9fa", foreground="#212529", font=("Segoe UI", 10))
        self.style.configure("TButton", background="#007bff", foreground="#ffffff", borderwidth=0, font=("Segoe UI", 10, "bold"))
        self.style.map("TButton", background=[("active", "#0056b3")])
        self.style.configure("TCombobox", fieldbackground="#ffffff", background="#e9ecef", foreground="#212529")
        
        # Load sensors
        self.sensors = self.load_sensors()
        self.camera_sources = [s["name"] for s in self.sensors]

        # Parameters State & Defaults
        default_source = self.camera_sources[0] if self.camera_sources else ""
        for src in self.camera_sources:
            if "Camera 1" in src:
                default_source = src
                break
        self.source_var = tk.StringVar(value=default_source)
        self.display_3d_var = tk.BooleanVar(value=True)
        self.invert_depth_var = tk.BooleanVar(value=False)
        
        # Sliders state variables
        self.depth_scale = tk.DoubleVar(value=8.0)
        self.grid_res = tk.IntVar(value=40)  # default 40x30 reconstruction grid
        self.blur_size = tk.IntVar(value=9)
        
        # Difference method state variables
        self.diff_method_var = tk.StringVar(value="Absolute Difference (ABS)")
        self.abs_thresh = tk.DoubleVar(value=0.08)
        self.abs_blur = tk.IntVar(value=9)
        self.grad_thresh = tk.DoubleVar(value=0.05)
        self.grad_ksize = tk.IntVar(value=5)
        self.lab_thresh = tk.DoubleVar(value=0.06)
        self.lab_wL = tk.DoubleVar(value=1.0)
        self.lab_wAB = tk.DoubleVar(value=1.0)
        
        # Otsu's Adaptive variables
        self.otsu_correct = tk.DoubleVar(value=1.0)
        
        # HSV variables
        self.hsv_thresh = tk.DoubleVar(value=0.08)
        self.hsv_wH = tk.DoubleVar(value=1.2)
        self.hsv_wS = tk.DoubleVar(value=0.8)
        self.hsv_wV = tk.DoubleVar(value=0.5)
        
        # Texture Contrast Difference (TCD) variables
        self.tcd_thresh = tk.DoubleVar(value=0.05)
        self.tcd_ksize = tk.IntVar(value=7)
        
        # Photometric stereo weights default
        self.w_xR = tk.DoubleVar(value=1.5)
        self.w_xG = tk.DoubleVar(value=-1.5)
        self.w_xB = tk.DoubleVar(value=0.0)
        self.w_yR = tk.DoubleVar(value=0.0)
        self.w_yG = tk.DoubleVar(value=0.0)
        self.w_yB = tk.DoubleVar(value=1.5)
        
        # Calibration state variables
        self.gain_R = tk.DoubleVar(value=1.0)
        self.gain_G = tk.DoubleVar(value=1.0)
        self.gain_B = tk.DoubleVar(value=1.0)
        self.crosstalk_R2B = tk.DoubleVar(value=0.0)
        self.crosstalk_G2B = tk.DoubleVar(value=0.0)
        self.bias_gx = tk.DoubleVar(value=0.0)
        self.bias_gy = tk.DoubleVar(value=0.0)
        self.detrend_kernel = tk.IntVar(value=0)  # 0 means disabled
        self.do_auto_calibrate = False
        
        # Vector and Object Calibration state variables
        self.vector_scale = tk.DoubleVar(value=2.5)         # Scale length for drawing vectors
        self.vector_min_mag = tk.DoubleVar(value=1.5)      # Noise gate for flow vector magnitude
        self.object_cutoff = tk.DoubleVar(value=0.0)       # Height cutoff for isolating object
        self.normalize_flow_var = tk.BooleanVar(value=False) # Contrast normalization toggle for flow

        # Multi-frame and Zeroing State
        self.bg_accum_frames_left = 0
        self.bg_accum_sum = None
        
        self.auto_calib_frames_left = 0
        self.auto_calib_gx_sum = None
        self.auto_calib_gy_sum = None
        
        self.set_zero_frames_left = 0
        self.set_zero_sum = None
        self.Z_zero = None
        self.status_var = tk.StringVar(value="Status: Ready")
        
        # Feature Toggles
        self.enable_heatmap_var = tk.BooleanVar(value=True)
        self.enable_flow_var = tk.BooleanVar(value=True)
        self.enable_reconstruction_var = tk.BooleanVar(value=True)

        self.custom_fields = []
        self.custom_field_vars = {}
        self.custom_fields_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "custom_fields.json")
        self.load_custom_fields()
        
        # Sequence Recording state
        self.capture_mode_var = tk.StringVar(value="Image")
        self.auto_capture_threshold = tk.IntVar(value=500)
        self.auto_capture_armed_var = tk.BooleanVar(value=False)
        self.is_recording_sequence = False
        self.current_sequence_dir = ""
        self.sequence_frame_counter = 0

        # Data Gathering state variables
        self.camera_res_var = tk.StringVar(value="Camera Native Resolution: Unknown")
        self.frame_scale_var = tk.DoubleVar(value=1.0)
        self.dataset_dir_var = tk.StringVar(value=os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")))
        self.label_var = tk.StringVar(value="")
        self.save_raw_var = tk.BooleanVar(value=True)
        self.save_contact_var = tk.BooleanVar(value=True)
        self.save_flow_var = tk.BooleanVar(value=True)
        self.save_height_3d_var = tk.BooleanVar(value=True)
        self.save_height_2d_var = tk.BooleanVar(value=True)
        self.save_mask_var = tk.BooleanVar(value=True)
        
        
        # Thread-safe frame caching
        self.current_frame = None
        self.current_heatmap = None
        self.current_deform = None
        self.current_flow = None
        self.current_height_2d = None
        
        # Popout window variables
        self.popout_window = None
        self.popout_canvas = None
        self.popout_fig = None
        self.popout_ax = None
        self.popout_surf = None
        self.surf = None
        
        # Camera / Stream parameters
        self.cap = None
        self.running = True
        self.ref_frame = None
        self.fps = 0.0
        self.gray_ref_cached = None
        
        # Create Layout
        self.create_widgets()
        
        # Load configuration if it exists
        self.load_config()
        

        
        # Initialize sample count display and labels list
        self.refresh_existing_labels()
        self.update_sample_count_display()
        
        # Initialize default source camera
        self.on_source_change(None)
        
        # Start processing loop in a daemon thread
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()
        
    def load_sensors(self):
        """Loads sensors from individual json files in sensor_configs folder."""
        import glob
        sensors = []
        configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        
        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir)
            # Migrate old sensors.json if it exists
            sensors_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sensors.json")
            if os.path.exists(sensors_path):
                try:
                    with open(sensors_path, "r") as f:
                        old_sensors = json.load(f)
                    for s in old_sensors:
                        name = s.pop("name", f"Sensor_{s.get('source', 0)}")
                        with open(os.path.join(configs_dir, f"{name}.json"), "w") as out_f:
                            json.dump(s, out_f, indent=4)
                except Exception as e:
                    print(f"Migration failed: {e}")
                    
        for conf_file in glob.glob(os.path.join(configs_dir, "*.json")):
            name = os.path.basename(conf_file).replace(".json", "")
            if name in ["config", "custom_fields"]:
                continue
            try:
                with open(conf_file, "r") as f:
                    s_data = json.load(f)
                s_data["name"] = name
                sensors.append(s_data)
            except Exception as e:
                print(f"Failed to load {conf_file}: {e}")


                    
        return sensors

    def save_sensors(self):
        import glob
        configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir)
            
        # Delete old files to keep directory in sync with self.sensors
        for f in glob.glob(os.path.join(configs_dir, "*.json")):
            name = os.path.basename(f).replace(".json", "")
            if name in ["config", "custom_fields"]:
                continue
            try:
                os.remove(f)
            except Exception:
                pass

        for s in self.sensors:
            try:
                s_copy = s.copy()
                name = s_copy.pop("name")
                with open(os.path.join(configs_dir, f"{name}.json"), "w") as f:
                    json.dump(s_copy, f, indent=4)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save sensor {s.get('name')}: {e}")

    def auto_scan_usb_cameras(self):
        """Scans for available USB cameras."""
        available = []
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(f"USB Camera {i}")
                cap.release()
        return available

    def open_sensor_manager(self):
        """Opens a dialog to manage sensors (CRUD)."""
        mgr = tk.Toplevel(self.root)
        mgr.title("Manage Sensors")
        mgr.geometry("400x420")
        mgr.configure(bg="#f8f9fa")
        mgr.grab_set()

        tk.Label(mgr, text="Configured Sensors", bg="#f8f9fa", font=("Segoe UI", 11, "bold")).pack(pady=(10, 5))

        list_frame = tk.Frame(mgr, bg="#f8f9fa")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        sensor_listbox = tk.Listbox(list_frame, font=("Segoe UI", 10))
        sensor_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=sensor_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sensor_listbox.configure(yscrollcommand=scrollbar.set)

        def refresh_list():
            sensor_listbox.delete(0, tk.END)
            for s in self.sensors:
                sensor_listbox.insert(tk.END, f"{s['name']} ({s['source']})")

        refresh_list()

        btn_frame = tk.Frame(mgr, bg="#f8f9fa")
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        def delete_sensor():
            sel = sensor_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            del self.sensors[idx]
            refresh_list()

        btn_action_frame = tk.Frame(btn_frame, bg="#f8f9fa")
        btn_action_frame.pack(fill=tk.X, pady=2)

        def populate_for_edit():
            sel = sensor_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            s = self.sensors[idx]
            
            name_ent.delete(0, tk.END)
            name_ent.insert(0, s["name"])
            
            src_val = str(s["source"])
            matched = False
            for val in src_ent["values"]:
                if val.startswith(f"{src_val} ("):
                    src_ent.set(val)
                    matched = True
                    break
            if not matched:
                src_ent.set(src_val)
                
            settings = {k: v for k, v in s.items() if k not in ("name", "source")}
            settings_ent.delete(0, tk.END)
            if settings:
                settings_ent.insert(0, json.dumps(settings))
                
            btn_add.config(text="Update Sensor", bg="#ffc107", fg="#212529", command=lambda: update_sensor(idx))

        tk.Button(btn_action_frame, text="Edit Selected", bg="#ffc107", fg="#212529", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=populate_for_edit).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(btn_action_frame, text="Delete Selected", bg="#dc3545", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=delete_sensor).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        add_frame = tk.LabelFrame(mgr, text="Add New Sensor", bg="#f8f9fa", font=("Segoe UI", 9, "bold"), padx=5, pady=5)
        add_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(add_frame, text="Name:", bg="#f8f9fa").grid(row=0, column=0, sticky=tk.W)
        name_ent = tk.Entry(add_frame)
        name_ent.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        tk.Label(add_frame, text="Source (Select):", bg="#f8f9fa").grid(row=1, column=0, sticky=tk.W)
        
        available = self.auto_scan_usb_cameras()
        formatted_sources = []
        for src in available:
            idx = src.split(" ")[-1]
            label = "Web Cam" if idx == "0" else src
            formatted_sources.append(f"{idx} ({label})")
            
        src_ent = ttk.Combobox(add_frame, values=formatted_sources, state="readonly")
        src_ent.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        
        tk.Label(add_frame, text="Settings JSON (opt):", bg="#f8f9fa").grid(row=2, column=0, sticky=tk.W)
        settings_ent = tk.Entry(add_frame)
        settings_ent.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        add_frame.columnconfigure(1, weight=1)

        def reset_add_form():
            name_ent.delete(0, tk.END)
            src_ent.set("")
            settings_ent.delete(0, tk.END)
            btn_add.config(text="Add Sensor", bg="#28a745", fg="white", command=add_sensor)

        def update_sensor(idx):
            name = name_ent.get().strip()
            src_val_str = src_ent.get().strip()
            settings_str = settings_ent.get().strip()
            if not name or not src_val_str:
                return
            try:
                src_val = int(src_val_str.split(" ")[0])
            except ValueError:
                src_val = src_val_str
                
            sensor_data = {"name": name, "source": src_val}
            if settings_str:
                try:
                    sensor_settings = json.loads(settings_str)
                    if isinstance(sensor_settings, dict):
                        sensor_data.update(sensor_settings)
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid JSON in settings: {e}")
                    return
                    
            self.sensors[idx] = sensor_data
            self.save_sensors()
            reset_add_form()
            refresh_list()

        def add_sensor():
            name = name_ent.get().strip()
            src_val_str = src_ent.get().strip()
            settings_str = settings_ent.get().strip()
            if not name or not src_val_str:
                return
            try:
                src_val = int(src_val_str.split(" ")[0])
            except ValueError:
                src_val = src_val_str
                
            sensor_data = {"name": name, "source": src_val}
            if settings_str:
                try:
                    sensor_settings = json.loads(settings_str)
                    if isinstance(sensor_settings, dict):
                        sensor_data.update(sensor_settings)
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid JSON in settings: {e}")
                    return
                    
            self.sensors.append(sensor_data)
            self.save_sensors()
            reset_add_form()
            refresh_list()

        btn_add = tk.Button(add_frame, text="Add Sensor", bg="#28a745", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=add_sensor)
        btn_add.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.EW)

        def save_and_close():
            self.save_sensors()
            self.refresh_sources()
            mgr.destroy()

        mgr.protocol("WM_DELETE_WINDOW", save_and_close)

        tk.Button(mgr, text="Save & Close", bg="#007bff", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, command=save_and_close).pack(fill=tk.X, padx=10, pady=(5, 10))

    def refresh_sources(self):
        """Reloads sensors from disk and updates UI dropdown."""
        self.sensors = self.load_sensors()
        self.camera_sources = [s["name"] for s in self.sensors]
        self.source_combo["values"] = self.camera_sources
        
        current_source = self.source_var.get()
        if current_source not in self.camera_sources:
            if self.camera_sources:
                self.source_var.set(self.camera_sources[0])
                self.on_source_change(None)
            else:
                self.source_var.set("")

    def create_widgets(self):
        # Master Frame
        main_frame = tk.Frame(self.root, bg="#f8f9fa")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- LEFT SIDEBAR: CONTAINER WITH SCROLLBAR ---
        sidebar_container = tk.Frame(main_frame, bg="#ffffff", width=380, bd=1, relief=tk.SOLID)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar_container.pack_propagate(False)
        
        # Title Label is static at the top of the container
        title_lbl = tk.Label(sidebar_container, text="DATASET CONTROLS", font=("Segoe UI", 13, "bold"), bg="#ffffff", fg="#007bff")
        title_lbl.pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        # Scrollable Canvas
        canvas = tk.Canvas(sidebar_container, bg="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar_container, orient=tk.VERTICAL, command=canvas.yview)
        
        scrollable_frame = tk.Frame(canvas, bg="#ffffff")
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create canvas window
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=300)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrolling components
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Mousewheel scroll binding
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Configure tab colors inside the scrollable area
        self.style.configure("TNotebook", background="#ffffff", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#e9ecef", foreground="#495057", padding=[8, 3], font=("Segoe UI", 9, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#007bff")])
        
        # Top-level notebook
        self.top_notebook = ttk.Notebook(scrollable_frame)
        self.top_notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create Data Gathering and Calibration frames
        tab_prediction = tk.Frame(self.top_notebook, bg="#ffffff")
        self.top_notebook.add(tab_prediction, text="Prediction")
        
        
        tab_calibration = tk.Frame(self.top_notebook, bg="#ffffff")
        self.top_notebook.add(tab_calibration, text="Settings")
        
        # Inner notebook (inside tab_calibration)
        notebook = ttk.Notebook(tab_calibration)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Tab 1: Setup & Filter
        tab_setup = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_setup, text="Setup")
        
        # Tab 2: Optical Flow Calib
        tab_flow = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_flow, text="Optical Flow")
        
        # Tab 3: Diff / Contact Calib
        tab_contact = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_contact, text="Contact")
        
        # Tab 4: Height Calib
        tab_height = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_height, text="Height")
        
        # --- PREDICTION TAB ---
        pred_title = tk.Label(tab_prediction, text="Inference Engine", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#007bff")
        pred_title.pack(anchor=tk.W, pady=(10, 5))

        # Model Selection
        tk.Label(tab_prediction, text="Select Model:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        self.model_var = tk.StringVar()
        self.cb_model = ttk.Combobox(tab_prediction, textvariable=self.model_var, values=list(MODEL_MODULES.keys()), state="readonly")
        if MODEL_MODULES:
            self.cb_model.current(0)
        self.cb_model.pack(fill=tk.X, pady=(2, 8))

        # Frame source
        tk.Label(tab_prediction, text="Input Frame Source:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        self.frame_source_var = tk.StringVar(value="Raw Frame")
        self.cb_frame_src = ttk.Combobox(tab_prediction, textvariable=self.frame_source_var, values=["Raw Frame", "Heatmap (2D Height)", "Flow", "Contact Mask"], state="readonly")
        self.cb_frame_src.pack(fill=tk.X, pady=(2, 8))

        # Toggle Continuous Prediction
        self.continuous_pred_var = tk.BooleanVar(value=False)
        tk.Checkbutton(tab_prediction, text="Continuous Prediction (Live)", variable=self.continuous_pred_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=5)

        # Prediction Threshold
        thresh_frame = tk.Frame(tab_prediction, bg="#ffffff")
        thresh_frame.pack(fill=tk.X, pady=2)
        tk.Label(thresh_frame, text="Prediction Threshold (px):", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.pred_threshold = tk.IntVar(value=100)
        tk.Entry(thresh_frame, textvariable=self.pred_threshold, font=("Segoe UI", 9), width=10, bg="#f8f9fa", relief=tk.FLAT).pack(side=tk.RIGHT)


        # Result Display
        self.pred_result_var = tk.StringVar(value="Waiting...")
        self.lbl_result = tk.Label(tab_prediction, textvariable=self.pred_result_var, font=("Segoe UI", 14, "bold"), bg="#007bff", fg="#ffffff", pady=10)
        self.lbl_result.pack(fill=tk.X, pady=10)

        # Variables for prediction loop
        self.last_pred_time = 0
        self.pred_interval = 0.5  # predict every 500ms when continuous

        # --- TAB 1: Setup ---
        tk.Label(tab_setup, text="Input Video Source:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        
        src_frame = tk.Frame(tab_setup, bg="#ffffff")
        src_frame.pack(fill=tk.X, pady=(2, 8))
        self.source_combo = ttk.Combobox(src_frame, textvariable=self.source_var, values=self.camera_sources, state="readonly")
        self.source_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.source_combo.bind("<<ComboboxSelected>>", self.on_source_change)
        
        tk.Label(tab_setup, textvariable=self.camera_res_var, bg="#ffffff", fg="#6c757d", font=("Segoe UI", 8, "italic")).pack(anchor=tk.W, pady=(0, 5))
        self.add_slider(tab_setup, "Frame Scale Factor:", self.frame_scale_var, 0.1, 1.0, is_int=False)
        
        self.btn_manage_sensors = tk.Button(src_frame, text="⚙ Manage Sensors", bg="#e9ecef", fg="#495057", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.open_sensor_manager)
        self.btn_manage_sensors.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.btn_refresh_sensors = tk.Button(src_frame, text="🔄 Refresh", bg="#e9ecef", fg="#495057", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.refresh_sources)
        self.btn_refresh_sensors.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Baseline Buttons
        btn_frame = tk.Frame(tab_setup, bg="#ffffff")
        btn_frame.pack(fill=tk.X, pady=5)
        self.btn_capture_ref = tk.Button(btn_frame, text="CAPTURE BACKGROUND", bg="#007bff", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.capture_reference, height=1)
        self.btn_capture_ref.pack(fill=tk.X, pady=(0, 5))
        self.btn_reset = tk.Button(btn_frame, text="RESET BASELINE", bg="#e9ecef", fg="#495057", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.reset_baseline, height=1)
        self.btn_reset.pack(fill=tk.X)
        
        self.add_slider(tab_setup, "Mesh Grid Res:", self.grid_res, 20, 80, is_int=True)
        self.add_slider(tab_setup, "Gaussian Filter Size:", self.blur_size, 1, 21, is_int=True)
        tk.Checkbutton(tab_setup, text="Enable 3D Render", variable=self.display_3d_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=5)
        
        # Feature Toggles UI
        tk.Checkbutton(tab_setup, text="Enable Contact Heatmap", variable=self.enable_heatmap_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=2)
        tk.Checkbutton(tab_setup, text="Enable Optical Flow", variable=self.enable_flow_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=2)
        tk.Checkbutton(tab_setup, text="Enable 3D Reconstruction", variable=self.enable_reconstruction_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=2)

        # Save Configuration Button
        # Save Toggles (Checkboxes)
        chk_frame = tk.LabelFrame(tab_setup, text="Select Data to Save", bg="#ffffff", fg="#007bff", font=("Segoe UI", 9, "bold"), padx=10, pady=8)
        chk_frame.pack(fill=tk.X, pady=(5, 10))

        tk.Checkbutton(chk_frame, text="Raw Feed Image (.png)", variable=self.save_raw_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=3)
        tk.Checkbutton(chk_frame, text="Contact Heatmap (.png)", variable=self.save_contact_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=3)
        tk.Checkbutton(chk_frame, text="Contact Mask (.png)", variable=self.save_mask_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=3)
        tk.Checkbutton(chk_frame, text="Optical Flow (visual & .npy)", variable=self.save_flow_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=3)
        tk.Checkbutton(chk_frame, text="3D Height Chart (.png)", variable=self.save_height_3d_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=3)
        tk.Checkbutton(chk_frame, text="2D Height Top View (.png)", variable=self.save_height_2d_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=3)

        self.btn_save_config = tk.Button(tab_setup, text="SAVE CONFIGURATION", bg="#ffc107", fg="#212529", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.save_config, height=1)
        self.btn_save_config.pack(fill=tk.X, pady=(10, 0))
        
        # --- TAB 2: Optical Flow ---
        flow_frame = tk.LabelFrame(tab_flow, text="Flow Parameters", bg="#ffffff", fg="#007bff", font=("Segoe UI", 9, "bold"), padx=5, pady=5)
        flow_frame.pack(fill=tk.X, pady=5)
        self.add_slider(flow_frame, "Vector Draw Scale:", self.vector_scale, 0.5, 10.0)
        self.add_slider(flow_frame, "Vector Gate (Min):", self.vector_min_mag, 0.01, 2.0)
        tk.Checkbutton(flow_frame, text="Normalize Flow Contrast", variable=self.normalize_flow_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=5)
        
        # --- TAB 3: Contact ---
        contact_frame = tk.LabelFrame(tab_contact, text="Contact Detection Method", bg="#ffffff", fg="#007bff", font=("Segoe UI", 9, "bold"), padx=5, pady=5)
        contact_frame.pack(fill=tk.X, pady=5)
        
        self.diff_combo = ttk.Combobox(contact_frame, textvariable=self.diff_method_var, values=[
            "Absolute Difference (ABS)", 
            "Gradient Magnitude (GRAD)", 
            "CIELAB Color Distance (LAB)",
            "Adaptive Otsu's Binarization (OTSU)",
            "HSV Color Shift (HSV)",
            "Texture Contrast Difference (TCD)"
        ], state="readonly")
        self.diff_combo.pack(fill=tk.X, pady=5)
        self.diff_combo.bind("<<ComboboxSelected>>", self.update_diff_widgets)
        
        # Dynamic slider frame
        self.diff_calib_frame = tk.Frame(tab_contact, bg="#ffffff")
        self.diff_calib_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Call initially to populate correct sliders
        self.update_diff_widgets(None)
        
        # --- TAB 4: Height ---
        self.add_slider(tab_height, "3D Depth Scale:", self.depth_scale, 0.5, 100.0)
        self.add_slider(tab_height, "Depth Detrending (HPF):", self.detrend_kernel, 0, 51, is_int=True)
        self.add_slider(tab_height, "Object Height Cutoff:", self.object_cutoff, 0.0, 5.0)
        tk.Checkbutton(tab_height, text="Invert 3D Depth", variable=self.invert_depth_var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=5)
        
        calib_btn_frame = tk.LabelFrame(tab_height, text="Calibration Actions", bg="#ffffff", fg="#007bff", font=("Segoe UI", 9, "bold"), padx=5, pady=5)
        calib_btn_frame.pack(fill=tk.X, pady=5)
        
        self.btn_autocal = tk.Button(calib_btn_frame, text="AUTO-CALIBRATE OFFSETS", bg="#28a745", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.trigger_auto_calibrate, height=1)
        self.btn_autocal.pack(fill=tk.X, pady=5)
        
        self.btn_set_zero = tk.Button(calib_btn_frame, text="SET HEIGHT ZERO", bg="#17a2b8", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.trigger_set_zero, height=1)
        self.btn_set_zero.pack(fill=tk.X, pady=5)
        
        self.btn_clear_zero = tk.Button(calib_btn_frame, text="CLEAR HEIGHT ZERO", bg="#6c757d", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.clear_zero, height=1)
        self.btn_clear_zero.pack(fill=tk.X, pady=5)
        
        # Gains Frame
        gains_frame = tk.LabelFrame(tab_height, text="Color Channel Gains", bg="#ffffff", fg="#007bff", font=("Segoe UI", 9, "bold"), padx=5, pady=5)
        gains_frame.pack(fill=tk.X, pady=5)
        self.add_mini_slider(gains_frame, "R Gain:", self.gain_R, 0.1, 10.0)
        self.add_mini_slider(gains_frame, "G Gain:", self.gain_G, 0.1, 10.0)
        self.add_mini_slider(gains_frame, "B Gain:", self.gain_B, 0.1, 10.0)
        
        # Crosstalk Frame
        ct_frame = tk.LabelFrame(tab_height, text="Crosstalk Subtraction (B-ch)", bg="#ffffff", fg="#007bff", font=("Segoe UI", 9, "bold"), padx=5, pady=5)
        ct_frame.pack(fill=tk.X, pady=5)
        self.add_mini_slider(ct_frame, "R -> B Crosstalk:", self.crosstalk_R2B, -1.0, 1.0)
        self.add_mini_slider(ct_frame, "G -> B Crosstalk:", self.crosstalk_G2B, -1.0, 1.0)
        
        # Gradient Biases Frame
        bias_frame = tk.LabelFrame(tab_height, text="Manual Gradient Offsets", bg="#ffffff", fg="#007bff", font=("Segoe UI", 9, "bold"), padx=5, pady=5)
        bias_frame.pack(fill=tk.X, pady=5)
        self.add_mini_slider(bias_frame, "Gx Bias:", self.bias_gx, -1.0, 1.0)
        self.add_mini_slider(bias_frame, "Gy Bias:", self.bias_gy, -1.0, 1.0)
        
        # Weights Frame
        pw_frame = tk.LabelFrame(tab_height, text="Gradient Weights (R, G, B)", bg="#ffffff", fg="#007bff", font=("Segoe UI", 9, "bold"), padx=5, pady=5)
        pw_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(pw_frame, text="Gx (Horizontal) Weights:", bg="#ffffff", font=("Segoe UI", 8, "bold"), fg="#6c757d").pack(anchor=tk.W, pady=(2, 0))
        self.add_mini_slider(pw_frame, "R weight:", self.w_xR, -3.0, 3.0)
        self.add_mini_slider(pw_frame, "G weight:", self.w_xG, -3.0, 3.0)
        self.add_mini_slider(pw_frame, "B weight:", self.w_xB, -3.0, 3.0)
        
        tk.Label(pw_frame, text="Gy (Vertical) Weights:", bg="#ffffff", font=("Segoe UI", 8, "bold"), fg="#6c757d").pack(anchor=tk.W, pady=(5, 0))
        self.add_mini_slider(pw_frame, "R weight:", self.w_yR, -3.0, 3.0)
        self.add_mini_slider(pw_frame, "G weight:", self.w_yG, -3.0, 3.0)
        self.add_mini_slider(pw_frame, "B weight:", self.w_yB, -3.0, 3.0)
        
        # Static status label below notebook, static at bottom
        self.status_lbl = tk.Label(sidebar_container, textvariable=self.status_var, font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#007bff", pady=5, bd=1, relief=tk.SUNKEN)
        self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
        
        self.fps_lbl = tk.Label(sidebar_container, text="FPS: 0.0", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#28a745", pady=5, bd=1, relief=tk.SUNKEN)
        self.fps_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
        
        # --- RIGHT AREA: 2X2 DASHBOARD GRID ---
        grid_frame = tk.Frame(main_frame, bg="#f8f9fa")
        grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure Grid Weights
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.rowconfigure(1, weight=1)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        
        # 1. Raw Stream Canvas
        self.p1 = tk.LabelFrame(grid_frame, text="Raw Feed", bg="#ffffff", fg="#007bff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        self.p1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.lbl_raw = tk.Label(self.p1, bg="#f8f9fa")
        self.lbl_raw.pack(fill=tk.BOTH, expand=True)
        
        # 2. Difference Map Canvas
        self.p2 = tk.LabelFrame(grid_frame, text="Difference / Contact Heatmap", bg="#ffffff", fg="#007bff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        self.p2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.lbl_diff = tk.Label(self.p2, bg="#f8f9fa")
        self.lbl_diff.pack(fill=tk.BOTH, expand=True)
        
        # 3. Deformation Vectors Canvas
        self.p3 = tk.LabelFrame(grid_frame, text="Deformation Field Vectors (Optical Flow)", bg="#ffffff", fg="#007bff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        self.p3.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.lbl_vectors = tk.Label(self.p3, bg="#f8f9fa")
        self.lbl_vectors.pack(fill=tk.BOTH, expand=True)
        
        # 4. 3D Mesh / Reconstruction (Matplotlib)
        self.p4 = tk.LabelFrame(grid_frame, text="3D Height Reconstruction", bg="#ffffff", fg="#007bff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        self.p4.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Pop-out Viewer Button
        self.btn_popout = tk.Button(self.p4, text="Pop-out Viewer ↗", bg="#007bff", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.open_popout_viewer, height=1)
        self.btn_popout.pack(fill=tk.X, pady=(0, 5))
        
        # Matplotlib Figure Embed (Light Theme)
        self.fig = Figure(figsize=(4.5, 3.5), dpi=100, facecolor="#ffffff")
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#ffffff")
        self.ax.view_init(elev=35, azim=45)
        # Clear axis ticks/grid to match theme
        self.ax.xaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
        self.ax.yaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
        self.ax.zaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
        self.ax.tick_params(colors='#212529', labelsize=8)
        self.ax.set_xlabel("X Grid", color="#212529", fontsize=8)
        self.ax.set_ylabel("Y Grid", color="#212529", fontsize=8)
        self.ax.set_zlabel("Height", color="#212529", fontsize=8)
        
        self.canvas_3d = FigureCanvasTkAgg(self.fig, master=self.p4)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def add_slider(self, parent, text, var, val_min, val_max, is_int=False):
        frame = tk.Frame(parent, bg="#ffffff")
        frame.pack(fill=tk.X, pady=(2, 6))
        tk.Label(frame, text=text, bg="#ffffff", fg="#495057", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        val_lbl = tk.Label(frame, text=f"{var.get()}", bg="#ffffff", fg="#007bff", font=("Segoe UI", 9, "bold"))
        val_lbl.pack(side=tk.RIGHT)
        
        def update_val(val):
            v = float(val)
            if is_int:
                v = int(np.round(v))
                var.set(v)
                val_lbl.config(text=f"{v}")
            else:
                var.set(np.round(v, 2))
                val_lbl.config(text=f"{v:.2f}")
                
        s = tk.Scale(parent, from_=val_min, to=val_max, resolution=0.01 if not is_int else 1, orient=tk.HORIZONTAL, variable=var, showvalue=False, command=update_val, bg="#ffffff", fg="#007bff", highlightthickness=0, troughcolor="#e9ecef")
        s.pack(fill=tk.X, pady=(0, 6))

    def add_mini_slider(self, parent, text, var, val_min, val_max, resolution=0.1):
        frame = tk.Frame(parent, bg="#ffffff")
        frame.pack(fill=tk.X, pady=1)
        tk.Label(frame, text=text, bg="#ffffff", fg="#6c757d", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        val_lbl = tk.Label(frame, text=f"{var.get():.2f}" if resolution < 0.1 else f"{var.get():.1f}", bg="#ffffff", fg="#212529", font=("Segoe UI", 8))
        val_lbl.pack(side=tk.RIGHT)
        
        def update_val(val):
            v = np.round(float(val), 2 if resolution < 0.1 else 1)
            var.set(v)
            val_lbl.config(text=f"{v:.2f}" if resolution < 0.1 else f"{v:.1f}")
            
        s = tk.Scale(parent, from_=val_min, to=val_max, resolution=resolution, orient=tk.HORIZONTAL, variable=var, showvalue=False, command=update_val, bg="#ffffff", highlightthickness=0, troughcolor="#e9ecef", width=10)
        s.pack(fill=tk.X, pady=(0, 2))

    def update_diff_widgets(self, event=None):
        """Dynamically builds sliders for the selected difference method."""
        # Clear previous sliders
        for widget in self.diff_calib_frame.winfo_children():
            widget.destroy()
            
        method = self.diff_method_var.get()
        if "Absolute Difference" in method:
            self.add_slider(self.diff_calib_frame, "ABS Threshold:", self.abs_thresh, 0.01, 0.5)
            self.add_slider(self.diff_calib_frame, "ABS Blur Size:", self.abs_blur, 1, 21, is_int=True)
        elif "Gradient Magnitude" in method:
            self.add_slider(self.diff_calib_frame, "GRAD Threshold:", self.grad_thresh, 0.01, 0.5)
            self.add_slider(self.diff_calib_frame, "GRAD Sobel Kernel:", self.grad_ksize, 3, 7, is_int=True)
        elif "CIELAB Color Distance" in method:
            self.add_slider(self.diff_calib_frame, "LAB Threshold:", self.lab_thresh, 0.01, 0.5)
            self.add_slider(self.diff_calib_frame, "LAB L-weight:", self.lab_wL, 0.0, 2.0)
            self.add_slider(self.diff_calib_frame, "LAB AB-weight:", self.lab_wAB, 0.0, 2.0)
        elif "Adaptive Otsu" in method:
            self.add_slider(self.diff_calib_frame, "Otsu Correct Mult:", self.otsu_correct, 0.5, 2.0)
        elif "HSV Color Shift" in method:
            self.add_slider(self.diff_calib_frame, "HSV Threshold:", self.hsv_thresh, 0.01, 0.5)
            self.add_slider(self.diff_calib_frame, "HSV H-weight:", self.hsv_wH, 0.0, 3.0)
            self.add_slider(self.diff_calib_frame, "HSV S-weight:", self.hsv_wS, 0.0, 3.0)
            self.add_slider(self.diff_calib_frame, "HSV V-weight:", self.hsv_wV, 0.0, 3.0)
        else: # Texture Contrast Difference
            self.add_slider(self.diff_calib_frame, "TCD Threshold:", self.tcd_thresh, 0.005, 0.3)
            self.add_slider(self.diff_calib_frame, "TCD Window size:", self.tcd_ksize, 3, 15, is_int=True)

    def save_config(self):
        """Saves current GUI settings to the active sensor config."""
        source_name = self.source_var.get()
        current_sensor_idx = None
        for i, s in enumerate(self.sensors):
            if s["name"] == source_name:
                current_sensor_idx = i
                break
                
        if current_sensor_idx is None:
            messagebox.showerror("Error", "No sensor selected to save settings to.")
            return

        settings = {
            "enable_heatmap": self.enable_heatmap_var.get(),
            "enable_flow": self.enable_flow_var.get(),
            "enable_reconstruction": self.enable_reconstruction_var.get(),
            "display_3d": self.display_3d_var.get(),
            "save_raw": self.save_raw_var.get(),
            "save_contact": self.save_contact_var.get(),
            "save_mask": self.save_mask_var.get(),
            "save_flow": self.save_flow_var.get(),
            "save_height_3d": self.save_height_3d_var.get(),
            "save_height_2d": self.save_height_2d_var.get(),
            "frame_scale": self.frame_scale_var.get(),
            "capture_mode": getattr(self, "capture_mode_var", tk.StringVar(value="Image")).get(),
            "auto_capture_threshold": getattr(self, "auto_capture_threshold", tk.IntVar(value=500)).get(),
            "invert_depth": self.invert_depth_var.get(),
            "depth_scale": self.depth_scale.get(),
            "grid_res": self.grid_res.get(),
            "blur_size": self.blur_size.get(),
            "diff_method": self.diff_method_var.get(),
            "abs_thresh": self.abs_thresh.get(),
            "abs_blur": self.abs_blur.get(),
            "grad_thresh": self.grad_thresh.get(),
            "grad_ksize": self.grad_ksize.get(),
            "lab_thresh": self.lab_thresh.get(),
            "lab_wL": self.lab_wL.get(),
            "lab_wAB": self.lab_wAB.get(),
            "otsu_correct": self.otsu_correct.get(),
            "hsv_thresh": self.hsv_thresh.get(),
            "hsv_wH": self.hsv_wH.get(),
            "hsv_wS": self.hsv_wS.get(),
            "hsv_wV": self.hsv_wV.get(),
            "tcd_thresh": self.tcd_thresh.get(),
            "tcd_ksize": self.tcd_ksize.get(),
            "vector_scale": self.vector_scale.get(),
            "vector_min_mag": self.vector_min_mag.get(),
            "normalize_flow": self.normalize_flow_var.get(),
            "object_cutoff": self.object_cutoff.get(),
            "gain_R": self.gain_R.get(),
            "gain_G": self.gain_G.get(),
            "gain_B": self.gain_B.get(),
            "crosstalk_R2B": self.crosstalk_R2B.get(),
            "crosstalk_G2B": self.crosstalk_G2B.get(),
            "bias_gx": self.bias_gx.get(),
            "bias_gy": self.bias_gy.get(),
            "detrend_kernel": self.detrend_kernel.get(),
            "w_xR": self.w_xR.get(),
            "w_xG": self.w_xG.get(),
            "w_xB": self.w_xB.get(),
            "w_yR": self.w_yR.get(),
            "w_yG": self.w_yG.get(),
            "w_yB": self.w_yB.get()
        }
        
        self.sensors[current_sensor_idx].update(settings)
        self.save_sensors()
        
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.json")
        try:
            with open(config_path, "w") as f:
                json.dump({"source": source_name}, f, indent=4)
            self.status_var.set("Status: Sensor Settings Saved")
            self.root.after(2000, lambda: self.status_var.set("Status: Ready"))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save last source config: {e}")


    def load_custom_fields(self):
        if os.path.exists(self.custom_fields_file):
            try:
                import json
                with open(self.custom_fields_file, "r") as f:
                    self.custom_fields = json.load(f)
            except Exception as e:
                print(f"Error loading custom fields: {e}")
                self.custom_fields = []
        else:
            self.custom_fields = []

    def save_custom_fields_config(self):
        try:
            import json
            with open(self.custom_fields_file, "w") as f:
                json.dump(self.custom_fields, f, indent=4)
        except Exception as e:
            print(f"Error saving custom fields: {e}")

    def render_custom_fields_ui(self):
        # Clear existing
        for widget in self.custom_fields_container.winfo_children():
            widget.destroy()
        self.custom_field_vars.clear()

        if not self.custom_fields:
            return

        for field in self.custom_fields:
            name = field["name"]
            ftype = field["type"]
            default = field.get("default", "")

            frame = tk.Frame(self.custom_fields_container, bg="#ffffff")
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=name + ":", bg="#ffffff", fg="#495057", font=("Segoe UI", 9), width=15, anchor=tk.W).pack(side=tk.LEFT)

            if ftype == "Text":
                var = tk.StringVar(value=default)
                ent = tk.Entry(frame, textvariable=var, font=("Segoe UI", 9), bg="#f1f3f5", relief=tk.FLAT)
                ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.custom_field_vars[name] = var
            elif ftype == "Dropdown":
                var = tk.StringVar(value=default)
                opts = field.get("options", [])
                cmb = ttk.Combobox(frame, textvariable=var, values=opts, state="readonly", font=("Segoe UI", 9))
                cmb.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.custom_field_vars[name] = var
            elif ftype == "Checkbox":
                var = tk.BooleanVar(value=bool(default))
                chk = tk.Checkbutton(frame, variable=var, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff")
                chk.pack(side=tk.LEFT)
                self.custom_field_vars[name] = var

    def open_custom_fields_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Configure Custom Fields")
        dlg.geometry("450x450")
        dlg.configure(bg="#ffffff")
        dlg.grab_set()

        list_frame = tk.Frame(dlg, bg="#ffffff")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        lb = tk.Listbox(list_frame, font=("Segoe UI", 9))
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        def refresh_list():
            lb.delete(0, tk.END)
            for f in self.custom_fields:
                lb.insert(tk.END, f"{f['name']} ({f['type']})")

        refresh_list()

        btn_rm = tk.Button(dlg, text="Remove Selected", command=lambda: remove_selected(), bg="#dc3545", fg="white", relief=tk.FLAT)
        btn_rm.pack(fill=tk.X, padx=10, pady=5)

        def remove_selected():
            sel = lb.curselection()
            if sel:
                self.custom_fields.pop(sel[0])
                refresh_list()

        add_frame = tk.LabelFrame(dlg, text="Add New Field", bg="#ffffff", padx=10, pady=10)
        add_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(add_frame, text="Name:", bg="#ffffff").grid(row=0, column=0, sticky=tk.W)
        name_var = tk.StringVar()
        tk.Entry(add_frame, textvariable=name_var).grid(row=0, column=1, sticky=tk.EW, pady=2)

        tk.Label(add_frame, text="Type:", bg="#ffffff").grid(row=1, column=0, sticky=tk.W)
        type_var = tk.StringVar(value="Text")
        ttk.Combobox(add_frame, textvariable=type_var, values=["Text", "Dropdown", "Checkbox"], state="readonly").grid(row=1, column=1, sticky=tk.EW, pady=2)

        tk.Label(add_frame, text="Choices (CSV):", bg="#ffffff").grid(row=2, column=0, sticky=tk.W)
        opts_var = tk.StringVar()
        tk.Entry(add_frame, textvariable=opts_var).grid(row=2, column=1, sticky=tk.EW, pady=2)

        def add_field():
            n = name_var.get().strip()
            t = type_var.get()
            if not n: return
            opts_str = opts_var.get().strip()
            opts = [o.strip() for o in opts_str.split(",") if o.strip()] if opts_str else []
            if t == "Dropdown" and not opts:
                from tkinter import messagebox
                messagebox.showwarning("Warning", "Please enter Choices (comma-separated) for the Dropdown.")
                return
            default = opts[0] if t == "Dropdown" and opts else ("" if t == "Text" else False)
            self.custom_fields.append({"name": n, "type": t, "options": opts, "default": default})
            name_var.set("")
            opts_var.set("")
            refresh_list()

        tk.Button(add_frame, text="Add Field", command=add_field, bg="#28a745", fg="white", relief=tk.FLAT).grid(row=3, column=0, columnspan=2, pady=10, sticky=tk.EW)

        def save_and_close():
            self.save_custom_fields_config()
            self.render_custom_fields_ui()
            dlg.destroy()

        tk.Button(dlg, text="Save & Close", command=save_and_close, bg="#007bff", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT).pack(fill=tk.X, padx=10, pady=10)

    def load_config(self):

        """Loads last used source from global config file."""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.json")
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path, "r") as f:
                settings = json.load(f)
            
            if "source" in settings:
                self.source_var.set(str(settings["source"]))
        except Exception as e:
            print(f"Failed to load global config: {e}")

    def on_source_change(self, event):
        """Called when user selects a different input source."""
        source_name = self.source_var.get()
        source_val = None
        current_sensor = None
        for s in self.sensors:
            if s["name"] == source_name:
                source_val = s["source"]
                current_sensor = s
                break
        
        if source_val is None:
            return

        # Release existing camera if active
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.ref_frame = None
        self.gray_ref_cached = None
        
        try:
            source_val = int(source_val)
        except ValueError:
            pass
            
        if isinstance(source_val, int):
            self.cap = cv2.VideoCapture(source_val, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Failed to open Camera {source_val}")
        else:
            self.cap = cv2.VideoCapture(source_val)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Failed to open IP Camera {source_name}")
                
        if self.cap and self.cap.isOpened() and current_sensor:
            if os.name == 'nt' or isinstance(source_val, int):
                # Minimize internal buffering to reduce latency lag
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
            # Apply dynamic camera settings if defined in sensor config
            target_w, target_h = 320, 240 # Default to gsrobotics native GelSight Mini resolution
            if "resolution" in current_sensor and isinstance(current_sensor["resolution"], (list, tuple)) and len(current_sensor["resolution"]) == 2:
                target_w, target_h = current_sensor["resolution"]
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            if "fps" in current_sensor:
                self.cap.set(cv2.CAP_PROP_FPS, current_sensor["fps"])
            if "exposure" in current_sensor:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, current_sensor["exposure"])
            if "brightness" in current_sensor:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, current_sensor["brightness"])
            if "contrast" in current_sensor:
                self.cap.set(cv2.CAP_PROP_CONTRAST, current_sensor["contrast"])
                
            # Load GUI settings stored in the sensor config
            def set_val(var, key, type_cast):
                if key in current_sensor:
                    var.set(type_cast(current_sensor[key]))

            set_val(self.enable_heatmap_var, "enable_heatmap", bool)
            set_val(self.enable_flow_var, "enable_flow", bool)
            set_val(self.enable_reconstruction_var, "enable_reconstruction", bool)
            set_val(self.display_3d_var, "display_3d", bool)
            set_val(self.save_raw_var, "save_raw", bool)
            set_val(self.save_contact_var, "save_contact", bool)
            set_val(self.save_mask_var, "save_mask", bool)
            set_val(self.save_flow_var, "save_flow", bool)
            set_val(self.save_height_3d_var, "save_height_3d", bool)
            set_val(self.save_height_2d_var, "save_height_2d", bool)
            set_val(self.frame_scale_var, "frame_scale", float)
            set_val(getattr(self, "capture_mode_var", tk.StringVar(value="Image")), "capture_mode", str)
            set_val(getattr(self, "auto_capture_threshold", tk.IntVar(value=500)), "auto_capture_threshold", int)
            
            # Refresh capture mode UI dynamically
            if hasattr(self, "on_capture_mode_change"):
                self.on_capture_mode_change()
            set_val(self.invert_depth_var, "invert_depth", bool)
            set_val(self.depth_scale, "depth_scale", float)
            set_val(self.grid_res, "grid_res", int)
            set_val(self.blur_size, "blur_size", int)
            set_val(self.diff_method_var, "diff_method", str)
            set_val(self.abs_thresh, "abs_thresh", float)
            set_val(self.abs_blur, "abs_blur", int)
            set_val(self.grad_thresh, "grad_thresh", float)
            set_val(self.grad_ksize, "grad_ksize", int)
            set_val(self.lab_thresh, "lab_thresh", float)
            set_val(self.lab_wL, "lab_wL", float)
            set_val(self.lab_wAB, "lab_wAB", float)
            set_val(self.otsu_correct, "otsu_correct", float)
            set_val(self.hsv_thresh, "hsv_thresh", float)
            set_val(self.hsv_wH, "hsv_wH", float)
            set_val(self.hsv_wS, "hsv_wS", float)
            set_val(self.hsv_wV, "hsv_wV", float)
            set_val(self.tcd_thresh, "tcd_thresh", float)
            set_val(self.tcd_ksize, "tcd_ksize", int)
            set_val(self.vector_scale, "vector_scale", float)
            set_val(self.vector_min_mag, "vector_min_mag", float)
            set_val(self.normalize_flow_var, "normalize_flow", bool)
            set_val(self.object_cutoff, "object_cutoff", float)
            set_val(self.gain_R, "gain_R", float)
            set_val(self.gain_G, "gain_G", float)
            set_val(self.gain_B, "gain_B", float)
            set_val(self.crosstalk_R2B, "crosstalk_R2B", float)
            set_val(self.crosstalk_G2B, "crosstalk_G2B", float)
            set_val(self.bias_gx, "bias_gx", float)
            set_val(self.bias_gy, "bias_gy", float)
            set_val(self.detrend_kernel, "detrend_kernel", int)
            set_val(self.w_xR, "w_xR", float)
            set_val(self.w_xG, "w_xG", float)
            set_val(self.w_xB, "w_xB", float)
            set_val(self.w_yR, "w_yR", float)
            set_val(self.w_yG, "w_yG", float)
            set_val(self.w_yB, "w_yB", float)
            
            # Rebuild dynamic sliders based on newly loaded method
            self.update_diff_widgets(None)
            
            # Automatically start background baseline calibration when camera connects
            self.capture_reference()
                
    def capture_reference(self):
        """Starts multi-frame background baseline calibration."""
        self.bg_accum_frames_left = 15
        self.bg_accum_sum = None
        self.gray_prev = None
        self.status_var.set("Status: Capturing BG...")
        
    def reset_baseline(self):
        """Clears reference frames and zero height baseline."""
        self.ref_frame = None
        self.gray_prev = None
        self.Z_zero = None
        self.status_var.set("Status: Baseline Reset")
        self.root.after(2000, lambda: self.status_var.set("Status: Ready"))

    def trigger_auto_calibrate(self):
        """Starts multi-frame gradient offset auto-calibration."""
        if self.ref_frame is None:
            messagebox.showwarning("Warning", "Please capture a background baseline first.")
            return
        self.auto_calib_frames_left = 15
        self.auto_calib_gx_sum = None
        self.auto_calib_gy_sum = None
        self.status_var.set("Status: Calibrating Offsets...")

    def trigger_set_zero(self):
        """Starts multi-frame zero-height calibration."""
        if self.ref_frame is None:
            messagebox.showwarning("Warning", "Please capture a background baseline first.")
            return
        self.set_zero_frames_left = 15
        self.set_zero_sum = None
        self.status_var.set("Status: Zeroing Height...")

    def clear_zero(self):
        """Clears the zero-height reference baseline."""
        self.Z_zero = None
        self.status_var.set("Status: Zero Height Cleared")
        self.root.after(2000, lambda: self.status_var.set("Status: Ready"))
        print("Zero-height reference cleared.")

    def open_popout_viewer(self):
        """Spawns an interactive, larger 3D visualization window."""
        if self.popout_window is not None:
            self.popout_window.lift()
            return
            
        self.popout_window = tk.Toplevel(self.root)
        self.popout_window.title("Dataset 3D Viewer")
        self.popout_window.geometry("800x650")
        self.popout_window.configure(bg="#ffffff")
        
        # Figure and 3D Axis Setup
        self.popout_fig = Figure(figsize=(8, 6.5), dpi=100, facecolor="#ffffff")
        self.popout_ax = self.popout_fig.add_subplot(111, projection='3d')
        self.popout_ax.set_facecolor("#ffffff")
        self.popout_ax.view_init(elev=35, azim=45)
        
        # Style
        self.popout_ax.xaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
        self.popout_ax.yaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
        self.popout_ax.zaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
        self.popout_ax.tick_params(colors='#212529', labelsize=9)
        self.popout_ax.set_xlabel("X Grid", color="#212529", fontsize=9)
        self.popout_ax.set_ylabel("Y Grid", color="#212529", fontsize=9)
        self.popout_ax.set_zlabel("Height", color="#212529", fontsize=10)
        
        # Embed canvas
        self.popout_canvas = FigureCanvasTkAgg(self.popout_fig, master=self.popout_window)
        self.popout_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Cleanup callback on close
        self.popout_window.protocol("WM_DELETE_WINDOW", self.close_popout_viewer)
        self.status_var.set("Status: Rendering in Pop-out")
        
    def close_popout_viewer(self):
        """Cleans up the pop-out window and returns rendering to dashboard."""
        if self.popout_window is not None:
            self.popout_window.destroy()
            self.popout_window = None
            self.popout_canvas = None
            self.popout_fig = None
            self.popout_ax = None

    # --- Video / Image Processing Thread Loop ---
    
    def run_prediction(self, frame):
        model_name = self.model_var.get()
        if not model_name:
            self.root.after(0, lambda: self.pred_result_var.set("No Model Selected"))
            return
            
        module_path = MODEL_MODULES.get(model_name)
        if not module_path:
            return
            
        try:
            mod = importlib.import_module(module_path)
            if hasattr(mod, "predict_frame"):
                result = mod.predict_frame(frame)
            else:
                result = "Model doesn't support predict_frame()"
            self.root.after(0, lambda r=result: self.pred_result_var.set(r))
        except Exception as e:
            self.root.after(0, lambda err=e: self.pred_result_var.set(f"Error: {err}"))

    def video_loop(self):
        prev_time = time.time()
        
        # Dense flow parameters
        lk_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
        while self.running:
            start_time = time.time()
            source = self.source_var.get()
            
            frame = None
            if self.cap is not None and self.cap.isOpened():
                ret, raw = self.cap.read()
                if ret:
                    frame = raw
                else:
                    time.sleep(0.01)
            
            if frame is not None:
                h_orig, w_orig = frame.shape[:2]
                scale = self.frame_scale_var.get()
                w_scaled, h_scaled = int(w_orig * scale), int(h_orig * scale)
                self.root.after(0, lambda w=w_orig, h=h_orig, ws=w_scaled, hs=h_scaled: self.camera_res_var.set(f"Resolution: {w}x{h} (Native)  ->  {ws}x{hs} (Scaled)"))
                if scale != 1.0 and scale > 0:
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
            if frame is None:
                # Live Stream Offline fallback placeholder
                if hasattr(self, 'ref_frame') and self.ref_frame is not None:
                    h, w = self.ref_frame.shape[:2]
                else:
                    h, w = 480, 640
                frame = np.zeros((h, w, 3), dtype=np.uint8) + 40 # Dark gray background
                
                # Active pulsing circle indicator
                cycle = int(time.time() * 15) % 360
                cx = int(w/2 + 80 * np.cos(np.radians(cycle)))
                cy = int(h/2 + 80 * np.sin(np.radians(cycle)))
                cv2.circle(frame, (cx, cy), 15, (0, 122, 255), -1)
                
                cv2.putText(frame, "Live Stream Offline", (max(10, w//2 - 160), max(30, h//2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, min(1.0, w/640.0), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Engine waiting for video connection...", (max(10, w//2 - 200), max(60, h//2 + 30)), cv2.FONT_HERSHEY_SIMPLEX, min(0.7, w/640.0), (200, 200, 200), 1, cv2.LINE_AA)
            
            h, w, _ = frame.shape
            
            # --- Capture Reference / Baseline Handling ---
            if hasattr(self, 'bg_accum_frames_left') and self.bg_accum_frames_left > 0:
                self.status_var.set(f"Status: Capturing BG [{16 - self.bg_accum_frames_left}/15]...")
                if self.bg_accum_sum is None:
                    self.bg_accum_sum = frame.astype(float)
                else:
                    self.bg_accum_sum += frame.astype(float)
                self.bg_accum_frames_left -= 1
                
                if self.ref_frame is None:
                    self.ref_frame = frame.copy()

                if self.bg_accum_frames_left == 0:
                    self.ref_frame = (self.bg_accum_sum / 15.0).astype(np.uint8)
                    self.bg_accum_sum = None
                    self.gray_prev = None
                    self.gray_ref_cached = None
                    self.Z_zero = None
                    self.status_var.set("Status: BG Captured")
                    self.root.after(2000, lambda: self.status_var.set("Status: Ready"))
                    print("Background baseline averaged from 15 frames.")
            elif self.ref_frame is None:
                # Default fallback if no reference exists
                self.ref_frame = frame.copy()
                self.gray_prev = None
                self.gray_ref_cached = None
            
            # Get background subtraction frame
            I_ref = self.ref_frame
            if I_ref is not None and I_ref.shape != frame.shape:
                I_ref = cv2.resize(I_ref, (frame.shape[1], frame.shape[0]))
                self.ref_frame = I_ref
                self.gray_ref_cached = None
                self.Z_zero = None # Z_zero shape also invalidated
            
            mask_cleaned = None
            # Compute difference based on the selected method ONLY if enabled
            if self.enable_heatmap_var.get():
                method = self.diff_method_var.get()
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if not hasattr(self, 'gray_ref_cached') or self.gray_ref_cached is None or self.gray_ref_cached.shape != gray_curr.shape:
                    self.gray_ref_cached = cv2.cvtColor(I_ref, cv2.COLOR_BGR2GRAY)
                    
                if "Absolute Difference" in method:
                    diff_gray = cv2.absdiff(gray_curr, self.gray_ref_cached)
                    abs_blur = self.abs_blur.get()
                    if abs_blur > 1:
                        abs_blur = abs_blur if abs_blur % 2 == 1 else abs_blur + 1
                        diff_gray = cv2.GaussianBlur(diff_gray, (abs_blur, abs_blur), 0)
                    thresh = int(self.abs_thresh.get() * 255)
                    _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
                elif "Gradient Magnitude" in method:
                    ksize = self.grad_ksize.get()
                    if ksize % 2 == 0: ksize += 1
                    
                    dx_curr = cv2.Sobel(gray_curr, cv2.CV_32F, 1, 0, ksize=ksize)
                    dy_curr = cv2.Sobel(gray_curr, cv2.CV_32F, 0, 1, ksize=ksize)
                    mag_curr = cv2.magnitude(dx_curr, dy_curr)
                    
                    dx_ref = cv2.Sobel(self.gray_ref_cached, cv2.CV_32F, 1, 0, ksize=ksize)
                    dy_ref = cv2.Sobel(self.gray_ref_cached, cv2.CV_32F, 0, 1, ksize=ksize)
                    mag_ref = cv2.magnitude(dx_ref, dy_ref)
                    
                    diff_grad = cv2.absdiff(mag_curr, mag_ref)
                    diff_gray = np.clip(diff_grad * 4, 0, 255).astype(np.uint8)
                    
                    thresh = int(self.grad_thresh.get() * 255)
                    _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
                elif "CIELAB Color Distance" in method:
                    lab_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(float)
                    lab_ref = cv2.cvtColor(I_ref, cv2.COLOR_BGR2Lab).astype(float)
                    
                    dL = lab_curr[:, :, 0] - lab_ref[:, :, 0]
                    da = lab_curr[:, :, 1] - lab_ref[:, :, 1]
                    db = lab_curr[:, :, 2] - lab_ref[:, :, 2]
                    
                    wL = self.lab_wL.get()
                    wAB = self.lab_wAB.get()
                    dist = np.sqrt((dL * wL)**2 + (da * wAB)**2 + (db * wAB)**2)
                    diff_gray = np.clip(dist * 2.5, 0, 255).astype(np.uint8)
                    
                    thresh = int(self.lab_thresh.get() * 255)
                    _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
                elif "Adaptive Otsu" in method:
                    diff_gray = cv2.absdiff(gray_curr, self.gray_ref_cached)
                    otsu_thresh_val, mask = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    corr = self.otsu_correct.get()
                    if corr != 1.0:
                        corrected_thresh = np.clip(otsu_thresh_val * corr, 1, 254)
                        _, mask = cv2.threshold(diff_gray, corrected_thresh, 255, cv2.THRESH_BINARY)
                elif "HSV Color Shift" in method:
                    hsv_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(float)
                    hsv_ref = cv2.cvtColor(I_ref, cv2.COLOR_BGR2HSV).astype(float)
                    
                    dH = np.abs(hsv_curr[:, :, 0] - hsv_ref[:, :, 0])
                    dH = np.minimum(dH, 180.0 - dH)
                    dS = np.abs(hsv_curr[:, :, 1] - hsv_ref[:, :, 1])
                    dV = np.abs(hsv_curr[:, :, 2] - hsv_ref[:, :, 2])
                    
                    wH = self.hsv_wH.get()
                    wS = self.hsv_wS.get()
                    wV = self.hsv_wV.get()
                    
                    dist = wH * dH + wS * dS + wV * dV
                    diff_gray = np.clip(dist * 1.5, 0, 255).astype(np.uint8)
                    
                    thresh = int(self.hsv_thresh.get() * 255)
                    _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
                else: 
                    ksize = self.tcd_ksize.get()
                    if ksize % 2 == 0: ksize += 1
                    def get_local_std(img):
                        f_img = img.astype(float)
                        mean_X = cv2.blur(f_img, (ksize, ksize))
                        mean_X2 = cv2.blur(f_img**2, (ksize, ksize))
                        var_X = mean_X2 - mean_X**2
                        return np.sqrt(np.clip(var_X, 0, None))
                    
                    std_curr = get_local_std(gray_curr)
                    std_ref = get_local_std(self.gray_ref_cached)
                    diff_std = np.abs(std_curr - std_ref)
                    diff_gray = np.clip(diff_std * 8, 0, 255).astype(np.uint8)
                    thresh = int(self.tcd_thresh.get() * 255)
                    _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)

                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel_open)
                
                diff_gray_scaled = np.zeros_like(diff_gray)
                mask_indices = mask_cleaned > 0
                if np.any(mask_indices):
                    max_val = np.max(diff_gray[mask_indices])
                    min_val = np.min(diff_gray[mask_indices])
                    if max_val > min_val:
                        diff_gray_scaled[mask_indices] = ((diff_gray[mask_indices].astype(float) - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                    else:
                        diff_gray_scaled[mask_indices] = 255
                
                heatmap = cv2.applyColorMap(diff_gray_scaled, cv2.COLORMAP_JET)
                alpha = cv2.GaussianBlur(mask_cleaned.astype(float) / 255.0, (15, 15), 0)
                alpha_3d = np.expand_dims(alpha, axis=2)
                bg_color = np.array([240, 240, 240], dtype=float)
                heatmap_blended = (heatmap.astype(float) * alpha_3d + bg_color * (1.0 - alpha_3d)).astype(np.uint8)
            else:
                heatmap_blended = None
            
            # Process difference channels for photometric stereo (still needed for height)
            if self.enable_reconstruction_var.get() or (hasattr(self, "auto_calib_frames_left") and self.auto_calib_frames_left > 0):
                dR = (frame[:, :, 2].astype(float) - I_ref[:, :, 2].astype(float)) / 255.0
                dG = (frame[:, :, 1].astype(float) - I_ref[:, :, 1].astype(float)) / 255.0
                dB = (frame[:, :, 0].astype(float) - I_ref[:, :, 0].astype(float)) / 255.0
                
                # Apply Independent Color Gains
                dR = dR * self.gain_R.get()
                dG = dG * self.gain_G.get()
                dB = dB * self.gain_B.get()
                
                # Apply Crosstalk Subtraction to the Blue channel (correcting R and G bleeding)
                dB = dB - self.crosstalk_R2B.get() * dR - self.crosstalk_G2B.get() * dG
                
                # Apply Gaussian Blur to gradients input
                b_size = self.blur_size.get()
                if b_size > 1:
                    b_size = b_size if b_size % 2 == 1 else b_size + 1
                    dR = cv2.GaussianBlur(dR, (b_size, b_size), 0)
                    dG = cv2.GaussianBlur(dG, (b_size, b_size), 0)
                    dB = cv2.GaussianBlur(dB, (b_size, b_size), 0)
                    
                # --- Photometric Stereo Gradient Mapping ---
                gx_full = self.w_xR.get() * dR + self.w_xG.get() * dG + self.w_xB.get() * dB
                gy_full = self.w_yR.get() * dR + self.w_yG.get() * dG + self.w_yB.get() * dB
                
                # --- 3D Surface Reconstruction (Full Resolution) ---
                # Handle Auto-Calibration request (Multi-frame averaged on full resolution)
                if hasattr(self, 'auto_calib_frames_left') and self.auto_calib_frames_left > 0:
                    self.status_var.set(f"Status: Calibrating Offsets [{16 - self.auto_calib_frames_left}/15]...")
                    if self.auto_calib_gx_sum is None:
                        self.auto_calib_gx_sum = gx_full.copy()
                        self.auto_calib_gy_sum = gy_full.copy()
                    else:
                        self.auto_calib_gx_sum += gx_full
                        self.auto_calib_gy_sum += gy_full
                    self.auto_calib_frames_left -= 1
                    
                    if self.auto_calib_frames_left == 0:
                        mean_gx = np.mean(self.auto_calib_gx_sum / 15.0)
                        mean_gy = np.mean(self.auto_calib_gy_sum / 15.0)
                        self.root.after(0, lambda gx_val=mean_gx, gy_val=mean_gy: (
                            self.bias_gx.set(-gx_val),
                            self.bias_gy.set(-gy_val)
                        ))
                        self.auto_calib_gx_sum = None
                        self.auto_calib_gy_sum = None
                        self.status_var.set("Status: Offsets Calibrated")
                        self.root.after(2000, lambda: self.status_var.set("Status: Ready"))
                        print(f"Gradient auto-calibration finished. Gx bias: {-mean_gx:.4f}, Gy bias: {-mean_gy:.4f}")
                    
                # Apply Gradient biases/offsets to full resolution gradients
                gx_full_biased = gx_full + self.bias_gx.get()
                gy_full_biased = gy_full + self.bias_gy.get()
                
                # Solve Poisson Equation at full resolution
                Z = solve_poisson_dst(gx_full_biased, gy_full_biased)
                
                # Normalize height scale relative to the downsampling factor to keep visual scale consistent.
                # Scipy's solver assumes unit spacing, meaning output amplitude scales with resolution.
                res = self.grid_res.get()
                norm_factor = w / res
                Z = Z / norm_factor
                
                # Handle Set Zero accumulation at full resolution
                if hasattr(self, 'set_zero_frames_left') and self.set_zero_frames_left > 0:
                    self.status_var.set(f"Status: Zeroing Height [{16 - self.set_zero_frames_left}/15]...")
                    if self.set_zero_sum is None:
                        self.set_zero_sum = Z.copy()
                    else:
                        self.set_zero_sum += Z
                    self.set_zero_frames_left -= 1
                    if self.set_zero_frames_left == 0:
                        self.Z_zero = self.set_zero_sum / 15.0
                        self.set_zero_sum = None
                        self.status_var.set("Status: Height Zeroed")
                        self.root.after(2000, lambda: self.status_var.set("Status: Ready"))
                        print("Zero-height reference baseline captured (averaged over 15 frames).")
                
                # Apply Zero-height subtraction if active
                if hasattr(self, 'Z_zero') and self.Z_zero is not None:
                    if self.Z_zero.shape == Z.shape:
                        Z = Z - self.Z_zero
                    else:
                        self.Z_zero = None
                
                # Apply scale factor
                scale = self.depth_scale.get()
                Z = Z * scale
                if self.invert_depth_var.get():
                    Z = -Z
                    
                # Apply Detrending (High-Pass Filter) at full resolution
                dk = self.detrend_kernel.get()
                if dk > 1:
                    dk = dk if dk % 2 == 1 else dk + 1
                    Z_low = cv2.GaussianBlur(Z, (dk, dk), 0)
                    Z = Z - Z_low
                    
                # Apply Object Shape Thresholding (Isolate Contact Object Height)
                cutoff = self.object_cutoff.get()
                if cutoff > 0:
                    mask_noise = np.abs(Z) < cutoff
                    Z[mask_noise] = 0.0
                    
                # Save final reconstructed Z mesh grid data for saving
                self.Z_mesh = Z
                
                # Downsample Z mesh for 3D GUI plotting to keep plotting responsive (~30 FPS)
                res = self.grid_res.get()
                grid_w = res
                grid_h = int(res * h / w)
                self.Z_plot = cv2.resize(Z, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
                self.X_mesh, self.Y_mesh = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
                
                # Create 2D Height top view colormap (using viridis matching the 3D surface plot)
                limit = scale * 0.5
                Z_norm = np.clip((Z + limit) / (2 * limit + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                height_2d_color = cv2.applyColorMap(Z_norm, cv2.COLORMAP_VIRIDIS)
                self.current_height_2d = height_2d_color.copy() if height_2d_color is not None else None
            else:
                self.Z_mesh = np.zeros((h, w), dtype=float)
                res = self.grid_res.get()
                grid_w = res
                grid_h = int(res * h / w)
                self.Z_plot = np.zeros((grid_h, grid_w), dtype=float)
                self.X_mesh, self.Y_mesh = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
                self.current_height_2d = np.zeros_like(frame)
            
            # --- Deformation Processing ---
            deform_frame = frame.copy()
            
            # Compute Dense Farneback Optical Flow (Used directly as silicone has no physical markers)
            flow = None
            if self.enable_flow_var.get():
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply contrast normalization if checked
                if self.normalize_flow_var.get():
                    gray_curr_proc = cv2.equalizeHist(gray_curr)
                else:
                    gray_curr_proc = gray_curr.copy()
                    
                if not hasattr(self, 'gray_prev') or self.gray_prev is None or self.gray_prev.shape != gray_curr.shape:
                    self.gray_prev = gray_curr_proc.copy()
                flow = cv2.calcOpticalFlowFarneback(self.gray_prev, gray_curr_proc, None, **lk_params)
                self.gray_prev = gray_curr_proc.copy()
                
                # Draw flow vector arrows
                step = 30
                vec_scale = self.vector_scale.get()
                min_mag = self.vector_min_mag.get()
                for y_coord in range(step // 2, h, step):
                    for x_coord in range(step // 2, w, step):
                        fx, fy = flow[y_coord, x_coord]
                        mag = np.sqrt(fx**2 + fy**2)
                        if mag > min_mag:  # calibrated threshold
                            p_start = (x_coord, y_coord)
                            p_end = (int(x_coord + fx * vec_scale), int(y_coord + fy * vec_scale)) # calibrated scale
                            cv2.arrowedLine(deform_frame, p_start, p_end, (0, 255, 255), 2, tipLength=0.4)
            
            # --- Update cache variables for data gathering ---
            self.current_frame = frame.copy() if frame is not None else None
            self.current_mask = mask_cleaned.copy() if mask_cleaned is not None else None
            self.current_contact_area = int(np.sum(mask_cleaned > 0)) if mask_cleaned is not None else 0
            self.current_heatmap = heatmap_blended.copy() if heatmap_blended is not None else None
            self.current_deform = deform_frame.copy() if deform_frame is not None else None
            self.current_flow = flow.copy() if flow is not None else None

            # --- Sequence Auto-Capture State Machine ---
            if getattr(self, 'is_armed', False) and getattr(self, 'capture_mode_var', None) and self.capture_mode_var.get() == "Video":
                try:
                    thresh = self.auto_capture_threshold.get()
                except Exception:
                    thresh = 9999999 # Safe fallback if input is empty
                if getattr(self, 'current_contact_area', 0) >= thresh:
                    if not self.is_recording_sequence:
                        # Start new sequence
                        label = self.label_var.get().strip()
                        if label:
                            base_dir_raw = self.dataset_dir_var.get().strip()
                            sensor_name = self.source_var.get().replace("/", "_").replace("\\", "_")
                            label_dir = os.path.join(base_dir_raw, sensor_name, label, "video")
                            self.current_sequence_dir = self.get_next_sequence_dir(label_dir)
                            os.makedirs(self.current_sequence_dir, exist_ok=True)
                            self.current_sequence_name = os.path.basename(self.current_sequence_dir)
                            self.is_recording_sequence = True
                            self.sequence_frame_counter = 0
                            
                            # UI indicator for recording
                            self.root.after(0, lambda: self.btn_arm_capture.config(bg="#ff0000", fg="white", text=f"RECORDING..."))

                    if self.is_recording_sequence and getattr(self, 'current_sequence_dir', ""):
                        # Save frame features silently
                        label = self.label_var.get().strip()
                        sensor_name = self.source_var.get().replace("/", "_").replace("\\", "_")
                        prefix = f"{self.current_sequence_name}_frame_{self.sequence_frame_counter:04d}_"
                        self.save_features_to_dir(self.current_sequence_dir, prefix=prefix, sensor_name=sensor_name, label=label)
                        self.sequence_frame_counter += 1
                else:
                    if self.is_recording_sequence:
                        # Stop sequence
                        self.is_recording_sequence = False
                        self.root.after(0, lambda: self.btn_arm_capture.config(bg="#dc3545", fg="white", text="DISARM (ARMED & WAITING...)"))
                        
                        short_dir = os.path.basename(os.path.dirname(self.current_sequence_dir)) + "/" + os.path.basename(self.current_sequence_dir)
                        self.root.after(0, lambda sd=short_dir: self.lbl_last_saved.config(text=f"Last Video: {sd}"))
                        self.root.after(0, self.refresh_existing_labels)
                        self.root.after(0, self.update_sample_count_display)

            # --- Trigger UI updates on the main thread (pass heatmap directly) ---
            
            # --- CONTINUOUS PREDICTION ---
            if getattr(self, 'continuous_pred_var', None) and self.continuous_pred_var.get():
                curr_t = time.time()
                if curr_t - getattr(self, 'last_pred_time', 0) >= getattr(self, 'pred_interval', 0.5):
                    self.last_pred_time = curr_t
                    src_type = self.frame_source_var.get()
                    pred_frame = None
                    if src_type == "Raw Frame":
                        pred_frame = frame
                    elif src_type == "Heatmap (2D Height)":
                        pred_frame = heatmap_blended
                    elif src_type == "Flow":
                        pred_frame = deform_frame
                    elif src_type == "Contact Mask":
                        pred_frame = mask_cleaned
                    
                    if pred_frame is not None:
                        try:
                            thresh = self.pred_threshold.get()
                        except Exception:
                            thresh = 100
                            
                        if getattr(self, 'current_contact_area', 0) >= thresh:
                            # Copy frame for thread safety
                            pf_copy = pred_frame.copy()
                            # Run in background thread to avoid blocking video loop
                            threading.Thread(target=self.run_prediction, args=(pf_copy,), daemon=True).start()
                        else:
                            self.root.after(0, lambda: self.pred_result_var.set("NO OBJECT"))


            self.root.after(0, self.update_ui_frames, frame, heatmap_blended, deform_frame)
            
            # FPS Calculation
            curr_time = time.time()
            self.fps = 1.0 / (curr_time - start_time + 1e-6)
            
            # Maintain processing loop rate
            time.sleep(max(0.005, 0.033 - (time.time() - start_time))) # aim for ~30 FPS

    def update_ui_frames(self, raw, diff, vectors):
        """Prepares images and triggers main frame rendering updates."""
        try:
            # Resize for layout panels
            panel_w, panel_h = 420, 310
            
            # Raw Stream Render
            raw_rgb = cv2.cvtColor(cv2.resize(raw, (panel_w, panel_h)), cv2.COLOR_BGR2RGB)
            raw_pil = ImageTk.PhotoImage(image=Image.fromarray(raw_rgb))
            self.lbl_raw.config(image=raw_pil)
            self.lbl_raw.image = raw_pil
            
            # Difference / Heatmap Render
            if diff is not None:
                heatmap_resized = cv2.resize(diff, (panel_w, panel_h))
                diff_rgb = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB)
                contact_area = getattr(self, 'current_contact_area', 0)
                text_color = (0, 255, 0) if contact_area > 0 else (200, 200, 200)
                cv2.putText(diff_rgb, f"Contact Area: {contact_area} px", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
            else:
                diff_rgb = np.zeros((panel_h, panel_w, 3), dtype=np.uint8) + 40
                cv2.putText(diff_rgb, "Feature Disabled", (panel_w//2 - 70, panel_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(diff_rgb, "(Saves Memory & CPU)", (panel_w//2 - 90, panel_h//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            diff_pil = ImageTk.PhotoImage(image=Image.fromarray(diff_rgb))
            self.lbl_diff.config(image=diff_pil)
            self.lbl_diff.image = diff_pil
            
            # Deformation Vectors Render
            if vectors is not None and self.enable_flow_var.get():
                vec_rgb = cv2.cvtColor(cv2.resize(vectors, (panel_w, panel_h)), cv2.COLOR_BGR2RGB)
            else:
                vec_rgb = np.zeros((panel_h, panel_w, 3), dtype=np.uint8) + 40
                cv2.putText(vec_rgb, "Feature Disabled", (panel_w//2 - 70, panel_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(vec_rgb, "(Saves Memory & CPU)", (panel_w//2 - 90, panel_h//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            vec_pil = ImageTk.PhotoImage(image=Image.fromarray(vec_rgb))
            self.lbl_vectors.config(image=vec_pil)
            self.lbl_vectors.image = vec_pil
            
            # 3D Matplotlib Render
            if not self.enable_reconstruction_var.get():
                if hasattr(self, 'surf') and self.surf is not None:
                    self.surf.remove()
                    self.surf = None
                if hasattr(self, 'popout_surf') and self.popout_surf is not None:
                    self.popout_surf.remove()
                    self.popout_surf = None
                self.ax.clear()
                self.ax.set_facecolor("#ffffff")
                self.ax.text2D(0.5, 0.5, "Feature Disabled\\n(Saves Memory & CPU)", transform=self.ax.transAxes, ha='center', va='center', color='#495057')
                self.canvas_3d.draw()
                if hasattr(self, 'popout_window') and self.popout_window is not None and hasattr(self, 'popout_ax') and self.popout_ax is not None:
                    self.popout_ax.clear()
                    self.popout_ax.set_facecolor("#ffffff")
                    self.popout_ax.text2D(0.5, 0.5, "Feature Disabled\\n(Saves Memory & CPU)", transform=self.popout_ax.transAxes, ha='center', va='center', color='#495057')
                    self.popout_canvas.draw()
            elif self.display_3d_var.get() and hasattr(self, 'Z_plot'):
                res = self.grid_res.get()
                grid_w = res
                grid_h = int(res * raw.shape[0] / raw.shape[1])
                limit = self.depth_scale.get() * 0.5
                
                # If Pop-out window is active, render there
                if self.popout_window is not None and hasattr(self, 'popout_ax') and self.popout_ax is not None:
                    elev = self.popout_ax.elev
                    azim = self.popout_ax.azim
                    
                    if hasattr(self, 'popout_surf') and self.popout_surf in self.popout_ax.collections:
                        self.popout_surf.remove()
                        
                    self.popout_surf = self.popout_ax.plot_surface(self.X_mesh, self.Y_mesh, self.Z_plot, cmap='viridis', edgecolor='none', shade=True)
                    self.popout_ax.set_xlim(0, grid_w - 1)
                    self.popout_ax.set_ylim(0, grid_h - 1)
                    self.popout_ax.set_zlim(-limit, limit)
                    try:
                        self.popout_ax.set_box_aspect((grid_w, grid_h, min(grid_w, grid_h) * 0.4))
                    except AttributeError:
                        pass
                    self.popout_ax.view_init(elev=elev, azim=azim)
                    self.popout_canvas.draw_idle()
                else:
                    # Otherwise render to the small embedded panel
                    elev = self.ax.elev
                    azim = self.ax.azim
                    
                    if hasattr(self, 'surf') and self.surf in self.ax.collections:
                        self.surf.remove()
                        
                    self.surf = self.ax.plot_surface(self.X_mesh, self.Y_mesh, self.Z_plot, cmap='viridis', edgecolor='none', shade=True)
                    self.ax.set_xlim(0, grid_w - 1)
                    self.ax.set_ylim(0, grid_h - 1)
                    self.ax.set_zlim(-limit, limit)
                    try:
                        self.ax.set_box_aspect((grid_w, grid_h, min(grid_w, grid_h) * 0.4))
                    except AttributeError:
                        pass
                    self.ax.view_init(elev=elev, azim=azim)
                    self.canvas_3d.draw_idle()
                
            # Update FPS labels
            self.fps_lbl.config(text=f"FPS: {self.fps:.1f}")
            
        except Exception as e:
            # Handle potential thread-safe closing states
            pass

    def browse_dataset_dir(self):
        """Opens a folder selection dialog to set the base dataset folder."""
        initial_dir = self.dataset_dir_var.get()
        if not os.path.exists(initial_dir):
            initial_dir = os.path.dirname(os.path.dirname(__file__))
        selected = filedialog.askdirectory(initial_dir=initial_dir, title="Select Base Dataset Directory")
        if selected:
            selected_abs = os.path.abspath(selected)
            self.dataset_dir_var.set(selected_abs)
            self.refresh_existing_labels()
            self.update_sample_count_display()

    def update_sample_count_display(self, event=None):
        pass

    def on_capture_mode_change(self):
        """Toggle UI elements based on selected mode."""
        mode = getattr(self, 'capture_mode_var', None)
        mode_str = mode.get() if mode else "Image"
        if mode_str == "Image":
            if hasattr(self, 'video_controls_frame'):
                self.video_controls_frame.pack_forget()
            if hasattr(self, 'btn_save_sample'):
                self.btn_save_sample.pack(fill=tk.X, pady=(5, 5))
            if hasattr(self, 'lbl_last_saved'):
                self.lbl_last_saved.pack(pady=2)
        else:
            if hasattr(self, 'btn_save_sample'):
                self.btn_save_sample.pack_forget()
            if hasattr(self, 'lbl_last_saved'):
                self.lbl_last_saved.pack_forget()
            if hasattr(self, 'video_controls_frame'):
                self.video_controls_frame.pack(fill=tk.X, pady=(5, 0))
            
        if hasattr(self, 'update_sample_count_display'):
            self.update_sample_count_display()

    def toggle_arm_capture(self):
        if not hasattr(self, 'is_armed'):
            self.is_armed = False
        self.is_armed = not self.is_armed
        
        if self.is_armed:
            self.capture_reference()
            self.btn_arm_capture.config(text="ARMED - WAITING FOR CONTACT", bg="#dc3545", fg="white")
        else:
            self.btn_arm_capture.config(text="ARM AUTO-CAPTURE", bg="#ffc107", fg="black")
            # If we disarm while recording, cut it off immediately
            if getattr(self, 'is_recording_sequence', False):
                self.is_recording_sequence = False
                self.current_sequence_dir = None
                self.status_var.set("Status: Sequence aborted by disarm.")


    def get_next_sequence_dir(self, label_dir):
        if not os.path.exists(label_dir):
            os.makedirs(label_dir, exist_ok=True)
            return os.path.join(label_dir, "sequence_001")
        existing = os.listdir(label_dir)
        seqs = [int(n.split("_")[1]) for n in existing if n.startswith("sequence_") and os.path.isdir(os.path.join(label_dir, n)) and len(n.split("_")) == 2 and n.split("_")[1].isdigit()]
        if not seqs:
            return os.path.join(label_dir, "sequence_001")
        return os.path.join(label_dir, f"sequence_{max(seqs) + 1:03d}")

    def get_next_file_idx(self, label_dir):
        if not os.path.exists(label_dir):
            os.makedirs(label_dir, exist_ok=True)
            return 0
        existing = os.listdir(label_dir)
        indices = []
        for name in existing:
            if name.endswith("_raw.png"):
                try:
                    idx = int(name.split("_")[0])
                    indices.append(idx)
                except ValueError:
                    pass
        return max(indices) + 1 if indices else 0

    def save_features_to_dir(self, sample_dir, prefix="", sensor_name="", label=""):
        os.makedirs(sample_dir, exist_ok=True)
        import cv2
        import numpy as np
        import time
        import json
        saved_files = []

        if self.save_raw_var.get() and self.current_frame is not None:
            raw_path = os.path.join(sample_dir, f"{prefix}raw.png")
            cv2.imwrite(raw_path, self.current_frame)
            saved_files.append(f"{prefix}raw.png")

        if self.save_contact_var.get() and self.current_heatmap is not None:
            contact_path = os.path.join(sample_dir, f"{prefix}contact_heatmap.png")
            cv2.imwrite(contact_path, self.current_heatmap)
            saved_files.append(f"{prefix}contact_heatmap.png")

        if hasattr(self, 'current_mask') and self.current_mask is not None:
            if getattr(self, 'save_mask_var', None) and getattr(self.save_mask_var, 'get', lambda: False)():
                mask_path = os.path.join(sample_dir, f"{prefix}contact_mask.png")
                cv2.imwrite(mask_path, self.current_mask)
                saved_files.append(f"{prefix}contact_mask.png")

        if self.save_flow_var.get() and hasattr(self, 'current_flow_img') and self.current_flow_img is not None:
            flow_path = os.path.join(sample_dir, f"{prefix}flow.png")
            cv2.imwrite(flow_path, self.current_flow_img)
            saved_files.append(f"{prefix}flow.png")
            if hasattr(self, 'current_flow_data'):
                npy_path = os.path.join(sample_dir, f"{prefix}flow_data.npy")
                np.save(npy_path, self.current_flow_data)
                saved_files.append(f"{prefix}flow_data.npy")

        if self.save_height_3d_var.get() and hasattr(self, 'current_height_img_3d') and self.current_height_img_3d is not None:
            h3d_path = os.path.join(sample_dir, f"{prefix}height_3d.png")
            cv2.imwrite(h3d_path, self.current_height_img_3d)
            saved_files.append(f"{prefix}height_3d.png")

        if self.save_height_2d_var.get() and hasattr(self, 'current_height_img_2d') and self.current_height_img_2d is not None:
            h2d_path = os.path.join(sample_dir, f"{prefix}height_2d.png")
            cv2.imwrite(h2d_path, self.current_height_img_2d)
            saved_files.append(f"{prefix}height_2d.png")

        # Save metadata
        safe_thresh = None
        try:
            safe_thresh = getattr(self, 'auto_capture_threshold').get() if hasattr(self, 'auto_capture_threshold') else None
        except Exception:
            safe_thresh = None

        meta = {
            "timestamp": time.time(),
            "sensor": sensor_name,
            "label": label,
            "capture_mode": getattr(self, 'capture_mode_var').get() if hasattr(self, 'capture_mode_var') else "Image",
            "diff_method": getattr(self, 'diff_method_var').get() if hasattr(self, 'diff_method_var') else "Unknown",
            "frame_scale": getattr(self, 'frame_scale_var').get() if hasattr(self, 'frame_scale_var') else 1.0,
            "contact_area_pixels": getattr(self, 'current_contact_area', 0),
            "auto_capture_threshold": safe_thresh,
            "is_video_sequence": getattr(self, 'is_recording_sequence', False),
            "sequence_name": getattr(self, 'current_sequence_name', None) if getattr(self, 'is_recording_sequence', False) else None,
            "saved_features": saved_files,
            "custom_fields": {k: v.get() for k, v in getattr(self, 'custom_field_vars', {}).items()}
        }
        
        meta_path = os.path.join(sample_dir, f"{prefix}metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    def save_data_point(self):
        """Saves current tactile features based on user selections."""
        label = self.label_var.get().strip()
        if not label:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "Please enter a Shape / Object Label before saving.")
            return

        base_dir_raw = self.dataset_dir_var.get().strip()
        if not base_dir_raw:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "Please specify a valid base dataset folder.")
            return

        sensor_name = self.source_var.get().replace("/", "_").replace("\\", "_")
        
        mode = self.capture_mode_var.get().lower()
        if self.current_frame is None:
            from tkinter import messagebox
            messagebox.showerror("Error", "No camera stream or offline frames are active yet.")
            return

        label_dir = os.path.join(base_dir_raw, sensor_name, label, mode)
        
        try:
            prefix = f"{self.get_next_file_idx(label_dir):04d}_"
            self.save_features_to_dir(label_dir, prefix=prefix, sensor_name=sensor_name, label=label)

            short_dir = os.path.basename(os.path.dirname(label_dir)) + "/" + os.path.basename(label_dir)
            self.lbl_last_saved.config(text=f"Last Saved: {short_dir}")
            self.refresh_existing_labels()
            self.update_sample_count_display()

            self.status_var.set(f"Status: Saved sample {prefix}raw.png!")
            self.root.after(3000, lambda: self.status_var.set("Status: Ready"))

        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to save data sample: {e}")

    def on_label_select(self, event=None):
        self.update_sample_count_display()

    def update_sample_count_display(self, event=None):
        pass

    def refresh_existing_labels(self):
        pass

    def on_close(self):
        """Cleans up resources and closes window."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
