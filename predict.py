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
import csv
import importlib
import sys
import pathlib

# Add the root directory to sys.path so 'models' can be found
_root_dir = os.path.dirname(os.path.abspath(__file__))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from algorithms.registry import get_algorithm_by_display_name, get_algorithm_specs

MODEL_MODULES = {
    "ResNet-18": "models.resnet18",
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
        self.root.title("TouchLab VTS")
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
        self.enable_raw_var = tk.BooleanVar(value=True)
        self.enable_heatmap_var = tk.BooleanVar(value=True)
        self.enable_flow_var = tk.BooleanVar(value=True)
        self.enable_reconstruction_var = tk.BooleanVar(value=True)
        self.layout_cols_var = tk.StringVar(value="Auto")

        self.custom_fields = []
        self.custom_field_vars = {}
        self.custom_fields_file = os.path.join(os.path.dirname(__file__), "config", "custom_fields.json")
        self.load_custom_fields()
        
        # Sequence Recording state
        self.capture_mode_var = tk.StringVar(value="Image")
        self.auto_capture_threshold = tk.IntVar(value=500)
        self.prediction_threshold = tk.IntVar(value=100)
        self.prediction_timer_seconds = tk.DoubleVar(value=3.0)
        self.auto_capture_armed_var = tk.BooleanVar(value=False)
        self.is_recording_sequence = False
        self.current_sequence_dir = ""
        self.sequence_frame_counter = 0

        # Data Gathering state variables
        self.camera_res_var = tk.StringVar(value="Camera Native Resolution: Unknown")
        self.frame_scale_var = tk.DoubleVar(value=1.0)
        self.dataset_dir_var = tk.StringVar(value=os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset")))
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
        configs_dir = os.path.join(os.path.dirname(__file__), "config")
        
        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir)
            # Migrate old sensors.json if it exists
            sensors_path = os.path.join(os.path.dirname(__file__), "sensors.json")
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
        configs_dir = os.path.join(os.path.dirname(__file__), "config")
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

    def refresh_camera(self):
        """Reload sensor definitions and reopen the currently selected camera."""
        self.status_var.set("Status: Reconnecting Camera...")
        previous_source = self.source_var.get().strip()
        self.refresh_sources()
        source_name = self.source_var.get().strip()
        if not source_name:
            self.status_var.set("Status: No Camera Selected")
            messagebox.showwarning("Camera Refresh", "No camera source is available.")
            return

        # refresh_sources already opens the first available source when selection changes.
        if source_name == previous_source:
            self.on_source_change(None)
        if self.cap is not None and self.cap.isOpened():
            self.log_message(f"Reconnected camera source: {source_name}")
        else:
            self.status_var.set("Status: Camera Disconnected")
            self.log_message(f"Failed to reconnect camera source: {source_name}")

    def show_main_view(self, view_name):
        if view_name not in self.main_views:
            return
        for name, frame in self.main_views.items():
            frame.pack_forget()
            button = self.main_nav_buttons.get(name)
            if button is not None:
                button.config(
                    bg="#ffffff",
                    fg="#495057",
                    relief=tk.FLAT,
                )
        self.main_views[view_name].pack(fill=tk.BOTH, expand=True)
        self.main_nav_buttons[view_name].config(
            bg="#007bff",
            fg="#ffffff",
            relief=tk.FLAT,
        )
        self.active_main_view = view_name
        if hasattr(self, "sidebar_canvas"):
            self.sidebar_canvas.yview_moveto(0)

    def open_settings_window(self):
        self.settings_window.deiconify()
        self.settings_window.lift()
        self.settings_window.focus_force()

    def hide_settings_window(self):
        self.settings_window.withdraw()

    def create_widgets(self):
        self.algorithm_specs = get_algorithm_specs()
        if not self.algorithm_specs:
            raise RuntimeError("No shape algorithms are registered")

        # Master Frame
        main_frame = tk.Frame(self.root, bg="#f8f9fa")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- LEFT SIDEBAR: CONTAINER WITH SCROLLBAR ---
        sidebar_container = tk.Frame(main_frame, bg="#ffffff", width=380, bd=1, relief=tk.SOLID)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar_container.pack_propagate(False)
        
        title_lbl = tk.Label(
            sidebar_container,
            text="TouchLab VTS",
            font=("Segoe UI", 14, "bold"),
            bg="#ffffff",
            fg="#212529",
        )
        title_lbl.pack(anchor=tk.W, padx=14, pady=(12, 8))

        nav_bar = tk.Frame(sidebar_container, bg="#e9ecef", padx=4, pady=4)
        nav_bar.pack(fill=tk.X, padx=10, pady=(0, 8))
        for column in range(4):
            nav_bar.columnconfigure(column, weight=1, uniform="main_nav")
        self.main_nav_buttons = {}
        for column, (name, label) in enumerate(
            (("predict", "Predict"), ("multi", "Multi Touch"), ("compare", "Compare"))
        ):
            button = tk.Button(
                nav_bar,
                text=label,
                bg="#ffffff",
                fg="#495057",
                activebackground="#007bff",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                bd=0,
                padx=5,
                pady=7,
                font=("Segoe UI", 9, "bold"),
                command=lambda selected=name: self.show_main_view(selected),
            )
            button.grid(row=0, column=column, sticky="ew", padx=1)
            self.main_nav_buttons[name] = button
        tk.Button(
            nav_bar,
            text="Settings",
            bg="#ffffff",
            fg="#495057",
            activebackground="#e9ecef",
            activeforeground="#212529",
            relief=tk.FLAT,
            bd=0,
            padx=5,
            pady=7,
            font=("Segoe UI", 9, "bold"),
            command=self.open_settings_window,
        ).grid(row=0, column=3, sticky="ew", padx=1)

        model_selector = tk.LabelFrame(
            sidebar_container,
            text="AI Model",
            bg="#ffffff",
            fg="#007bff",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=7,
        )
        model_selector.pack(fill=tk.X, padx=10, pady=(0, 8))
        model_selector.columnconfigure(1, weight=1)
        self.arch_var = tk.StringVar(value="ResNet-18")
        self.model_var = tk.StringVar()
        tk.Label(
            model_selector,
            text="Architecture",
            bg="#ffffff",
            fg="#495057",
            font=("Segoe UI", 8),
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 5))
        self.cb_arch = ttk.Combobox(
            model_selector,
            textvariable=self.arch_var,
            values=["ResNet-18"],
            state="readonly",
            width=15,
        )
        self.cb_arch.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 5))
        self.cb_arch.bind("<<ComboboxSelected>>", self.on_model_selected)
        tk.Label(
            model_selector,
            text="Weights",
            bg="#ffffff",
            fg="#495057",
            font=("Segoe UI", 8),
        ).grid(row=1, column=0, sticky=tk.W, padx=(0, 8))
        tk.Entry(
            model_selector,
            textvariable=self.model_var,
            state="readonly",
            bg="#f8f9fa",
            relief=tk.FLAT,
            font=("Segoe UI", 8),
        ).grid(row=1, column=1, sticky="ew")
        tk.Button(
            model_selector,
            text="Browse",
            bg="#e9ecef",
            fg="#495057",
            relief=tk.FLAT,
            bd=0,
            command=self.browse_weights,
            width=8,
        ).grid(row=1, column=2, padx=(6, 0))
        
        # Scrollable Canvas
        canvas = tk.Canvas(sidebar_container, bg="#ffffff", highlightthickness=0)
        self.sidebar_canvas = canvas
        scrollbar = ttk.Scrollbar(sidebar_container, orient=tk.VERTICAL, command=canvas.yview)
        
        scrollable_frame = tk.Frame(canvas, bg="#ffffff")
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create canvas window
        canvas_window = canvas.create_window(
            (0, 0), window=scrollable_frame, anchor="nw"
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(canvas_window, width=event.width),
        )
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
        
        tab_prediction = tk.Frame(scrollable_frame, bg="#ffffff")
        tab_multi_prediction = tk.Frame(scrollable_frame, bg="#ffffff")
        tab_comparison = tk.Frame(scrollable_frame, bg="#ffffff")
        self.tab_multi_prediction = tab_multi_prediction
        self.tab_comparison = tab_comparison

        self.main_views = {
            "predict": tab_prediction,
            "multi": tab_multi_prediction,
            "compare": tab_comparison,
        }
        self.active_main_view = "predict"

        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("TouchLab VTS Settings")
        self.settings_window.geometry("780x760")
        self.settings_window.minsize(680, 620)
        self.settings_window.configure(bg="#f8f9fa")
        self.settings_window.transient(self.root)
        self.settings_window.protocol("WM_DELETE_WINDOW", self.hide_settings_window)
        self.settings_window.withdraw()

        settings_header = tk.Frame(self.settings_window, bg="#ffffff")
        settings_header.pack(fill=tk.X)
        tk.Label(
            settings_header,
            text="Settings",
            font=("Segoe UI", 16, "bold"),
            bg="#ffffff",
            fg="#212529",
            padx=18,
            pady=14,
        ).pack(side=tk.LEFT)

        settings_footer = tk.Frame(self.settings_window, bg="#ffffff", padx=14, pady=10)
        settings_footer.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(
            settings_footer,
            textvariable=self.status_var,
            bg="#ffffff",
            fg="#6c757d",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

        notebook = ttk.Notebook(self.settings_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=(12, 8))
        
        # Tab 1: Setup & Filter
        tab_setup = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_setup, text="Setup")
        
        # Tab 2: Optical Flow Calib
        tab_flow = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_flow, text="Optical Flow")
        
        # Tab 3: Diff / Contact Calib
        tab_contact = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_contact, text="Contact")

        # Tab 4: Prediction capture
        tab_prediction_settings = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_prediction_settings, text="Prediction")

        # Tab 5: Shape algorithm configurations
        tab_algorithm_settings = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_algorithm_settings, text="Algorithms")
        
        # Tab 6: Height Calib
        tab_height = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_height, text="Height")

        prediction_capture_frame = tk.LabelFrame(
            tab_prediction_settings,
            text="Touch Capture",
            bg="#ffffff",
            fg="#007bff",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=8,
        )
        prediction_capture_frame.pack(fill=tk.X, pady=8)
        tk.Label(
            prediction_capture_frame,
            text="Prediction Threshold (px):",
            bg="#ffffff",
            fg="#495057",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, sticky=tk.W, pady=3)
        tk.Entry(
            prediction_capture_frame,
            textvariable=self.prediction_threshold,
            font=("Segoe UI", 9),
            width=10,
            bg="#f8f9fa",
            relief=tk.FLAT,
        ).grid(row=0, column=1, padx=8, pady=3, sticky=tk.W)
        tk.Label(
            prediction_capture_frame,
            text="Touch Timer (s):",
            bg="#ffffff",
            fg="#495057",
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, sticky=tk.W, pady=3)
        tk.Entry(
            prediction_capture_frame,
            textvariable=self.prediction_timer_seconds,
            font=("Segoe UI", 9),
            width=10,
            bg="#f8f9fa",
            relief=tk.FLAT,
        ).grid(row=1, column=1, padx=8, pady=3, sticky=tk.W)

        algorithm_config_frame = tk.LabelFrame(
            tab_algorithm_settings,
            text="Algorithm Configuration Files",
            bg="#ffffff",
            fg="#007bff",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=10,
        )
        algorithm_config_frame.pack(fill=tk.X, pady=8)
        algorithm_config_frame.columnconfigure(1, weight=1)
        tk.Label(
            algorithm_config_frame,
            text="Leave a path empty to use the selected model's sidecar, then the bundled default.",
            bg="#ffffff",
            fg="#6c757d",
            font=("Segoe UI", 8),
            wraplength=680,
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 9))
        scan_frame = tk.Frame(algorithm_config_frame, bg="#ffffff")
        scan_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        scan_frame.columnconfigure(1, weight=1)
        self.optimization_run_var = tk.StringVar(value="No optimization run selected")
        tk.Button(
            scan_frame,
            text="Scan Optimization Run",
            bg="#007bff",
            fg="#ffffff",
            activebackground="#0056b3",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            command=self.scan_optimization_run,
            padx=10,
            pady=5,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            scan_frame,
            textvariable=self.optimization_run_var,
            bg="#ffffff",
            fg="#6c757d",
            font=("Segoe UI", 8),
            anchor=tk.W,
        ).grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.algorithm_config_vars = {}
        for row, spec in enumerate(self.algorithm_specs, start=2):
            variable = tk.StringVar(value="")
            self.algorithm_config_vars[spec.key] = variable
            tk.Label(
                algorithm_config_frame,
                text=spec.display_name,
                bg="#ffffff",
                fg="#495057",
                font=("Segoe UI", 9, "bold"),
                anchor=tk.W,
            ).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 8))
            tk.Entry(
                algorithm_config_frame,
                textvariable=variable,
                state="readonly",
                readonlybackground="#f8f9fa",
                relief=tk.FLAT,
                font=("Segoe UI", 8),
            ).grid(row=row, column=1, sticky="ew", pady=4)
            tk.Button(
                algorithm_config_frame,
                text="Browse",
                bg="#e9ecef",
                fg="#495057",
                relief=tk.FLAT,
                bd=0,
                width=8,
                command=lambda selected=spec: self.browse_algorithm_config(selected),
            ).grid(row=row, column=2, padx=(6, 3), pady=4)
            tk.Button(
                algorithm_config_frame,
                text="Auto",
                bg="#ffffff",
                fg="#495057",
                relief=tk.FLAT,
                bd=1,
                width=6,
                command=lambda selected=spec: self.clear_algorithm_config(selected),
            ).grid(row=row, column=3, padx=(3, 0), pady=4)
        
        # --- PREDICTION TAB ---
        pred_title = tk.Label(tab_prediction, text="Inference Engine", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#007bff")
        pred_title.pack(anchor=tk.W, pady=(10, 5))

        # Frame source
        tk.Label(tab_prediction, text="Input Frame Source:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        self.frame_source_var = tk.StringVar(value="Raw Frame")
        self.cb_frame_src = ttk.Combobox(tab_prediction, textvariable=self.frame_source_var, values=["Raw Frame", "Heatmap (2D Height)", "Flow", "Contact Mask"], state="readonly")
        self.cb_frame_src.pack(fill=tk.X, pady=(2, 8))

        # Result Display
        self.pred_result_var = tk.StringVar(value="Waiting...")
        self.lbl_result = tk.Label(tab_prediction, textvariable=self.pred_result_var, font=("Segoe UI", 16, "bold"), bg="#007bff", fg="#ffffff", pady=15)
        self.lbl_result.pack(fill=tk.X, pady=(10, 5))

        # Class Probabilities Table
        tk.Label(tab_prediction, text="Class Probabilities:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        self.tree_probs = ttk.Treeview(tab_prediction, columns=("Class", "Probability"), show="headings", height=5)
        self.tree_probs.heading("Class", text="Class")
        self.tree_probs.heading("Probability", text="Probability (%)")
        self.tree_probs.column("Class", width=120)
        self.tree_probs.column("Probability", width=100, anchor=tk.E)
        self.tree_probs.pack(fill=tk.X, pady=(0, 10))
        
        self.root.after(100, self.on_model_selected)

        # Logs Section
        tk.Label(tab_prediction, text="Logs:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        self.txt_logs = tk.Text(tab_prediction, height=8, bg="#f8f9fa", font=("Consolas", 8), state=tk.DISABLED)
        self.txt_logs.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- MULTI TOUCH PREDICTION TAB ---
        multi_title = tk.Label(tab_multi_prediction, text="Multi Touch Inference", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#28a745")
        multi_title.pack(anchor=tk.W, pady=(10, 5))

        # Shape Algorithm Selection
        tk.Label(tab_multi_prediction, text="Shape Algorithm:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        algorithm_names = [spec.display_name for spec in self.algorithm_specs]
        self.shape_algo_var = tk.StringVar(value=algorithm_names[0])
        self.cb_shape_algo = ttk.Combobox(
            tab_multi_prediction,
            textvariable=self.shape_algo_var,
            values=algorithm_names,
            state="readonly",
        )
        self.cb_shape_algo.pack(fill=tk.X, pady=(2, 8))
        self.cb_shape_algo.bind("<<ComboboxSelected>>", self.on_shape_algo_selected)
        self.shape_algo_spec = self.algorithm_specs[0]
        self.shape_algo = self.create_shape_algorithm(self.shape_algo_spec)

        # Status Display
        self.multi_status_var = tk.StringVar(value="Waiting for Object...")
        self.lbl_multi_status = tk.Label(tab_multi_prediction, textvariable=self.multi_status_var, font=("Segoe UI", 12, "bold"), bg="#e9ecef", fg="#495057", pady=10, wraplength=280)
        self.lbl_multi_status.pack(fill=tk.X, pady=(10, 5))

        # Global Shape Prediction
        self.global_shape_var = tk.StringVar(value="Global Shape: Unknown (0%)")
        self.lbl_global_shape = tk.Label(tab_multi_prediction, textvariable=self.global_shape_var, font=("Segoe UI", 12, "bold"), bg="#007bff", fg="#ffffff", pady=10, wraplength=280)
        self.lbl_global_shape.pack(fill=tk.X, pady=(5, 5))

        # Trial action buttons
        multi_actions = tk.Frame(tab_multi_prediction, bg="#ffffff")
        multi_actions.pack(fill=tk.X, pady=(5, 10))
        self.btn_reset_multi = tk.Button(multi_actions, text="Reset Trial", bg="#dc3545", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, command=self.reset_multi_trial)
        self.btn_reset_multi.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_export_multi = tk.Button(multi_actions, text="Export CSV", bg="#198754", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, command=self.export_multi_touch_csv)
        self.btn_export_multi.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Touch History Table
        tk.Label(tab_multi_prediction, text="Touch History:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        self.tree_multi = ttk.Treeview(
            tab_multi_prediction,
            columns=("Touch", "Feature", "ShapeConfidence"),
            show="headings",
            height=8,
        )
        self.tree_multi.heading("Touch", text="Touch #")
        self.tree_multi.heading("Feature", text="Local Feature")
        self.tree_multi.heading("ShapeConfidence", text="Global Shape Confidence")
        self.tree_multi.column("Touch", width=58, anchor=tk.CENTER, stretch=False)
        self.tree_multi.column("Feature", width=135, anchor=tk.W, stretch=True)
        self.tree_multi.column("ShapeConfidence", width=170, anchor=tk.W, stretch=True)
        self.tree_multi.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # State Variables
        self.multi_timer_active = False
        self.multi_timer_start = 0
        self.multi_touch_counter = 0
        self.multi_waiting_for_removal = False
        self.multi_trial_stopped = False
        self.multi_touch_history = []

        # --- SHAPE COMPARISON TAB ---
        compare_title = tk.Label(tab_comparison, text="Compare Shape Algorithms", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#6f42c1")
        compare_title.pack(anchor=tk.W, pady=(10, 5))

        tk.Label(tab_comparison, text="Compare Algorithms:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(5, 0))
        compare_checks = tk.Frame(tab_comparison, bg="#ffffff")
        compare_checks.pack(fill=tk.X, pady=(1, 6))
        self.compare_algorithm_vars = {}
        for spec in self.algorithm_specs:
            enabled_var = tk.BooleanVar(value=True)
            self.compare_algorithm_vars[spec.key] = enabled_var
            tk.Checkbutton(
                compare_checks,
                text=spec.display_name,
                variable=enabled_var,
                command=self.reset_compare_trial,
                bg="#ffffff",
                fg="#212529",
                selectcolor="#ffffff",
                activebackground="#ffffff",
                activeforeground="#212529",
            ).pack(anchor=tk.W)

        self.compare_status_var = tk.StringVar(value="Waiting for Object...")
        compare_action_bar = tk.Frame(tab_comparison, bg="#ffffff")
        compare_action_bar.pack(fill=tk.X, pady=(8, 5))
        self.lbl_compare_status = tk.Label(
            compare_action_bar,
            textvariable=self.compare_status_var,
            font=("Segoe UI", 9, "bold"),
            bg="#e9ecef",
            fg="#495057",
            pady=7,
            padx=8,
            anchor=tk.W,
        )
        self.lbl_compare_status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            compare_action_bar,
            text="Reset",
            bg="#dc3545",
            fg="white",
            activebackground="#bb2d3b",
            activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            bd=0,
            command=self.reset_compare_trial,
            padx=12,
            pady=7,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        results_frame = tk.LabelFrame(
            tab_comparison,
            text="Current Results",
            bg="#ffffff",
            fg="#007bff",
            font=("Segoe UI", 9, "bold"),
            padx=5,
            pady=5,
        )
        results_frame.pack(fill=tk.X, pady=(5, 8))
        self.tree_compare = ttk.Treeview(
            results_frame,
            columns=("Algorithm", "Prediction", "Confidence"),
            show="headings",
            height=4,
        )
        self.tree_compare.heading("Algorithm", text="Algorithm")
        self.tree_compare.heading("Prediction", text="Shape")
        self.tree_compare.heading("Confidence", text="Confidence")
        self.tree_compare.column("Algorithm", width=105, minwidth=80, anchor=tk.W, stretch=True)
        self.tree_compare.column("Prediction", width=120, minwidth=90, anchor=tk.W, stretch=True)
        self.tree_compare.column("Confidence", width=90, minwidth=75, anchor=tk.E, stretch=False)
        self.tree_compare.tag_configure("accepted", foreground="#198754")
        self.tree_compare.tag_configure("uncertain", foreground="#b35c00")
        self.tree_compare.tag_configure("active", foreground="#495057")
        self.tree_compare.pack(fill=tk.X)
        self.compare_result_summary_var = tk.StringVar(value="Waiting for first touch")
        tk.Label(
            results_frame,
            textvariable=self.compare_result_summary_var,
            bg="#f8f9fa",
            fg="#495057",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
            padx=8,
            pady=6,
        ).pack(fill=tk.X, pady=(5, 0))

        details_bar = tk.Frame(tab_comparison, bg="#ffffff")
        details_bar.pack(fill=tk.X, pady=(5, 10))
        self.compare_details_summary_var = tk.StringVar(value="No details recorded")
        tk.Label(
            details_bar,
            textvariable=self.compare_details_summary_var,
            bg="#ffffff",
            fg="#6c757d",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT)
        tk.Button(
            details_bar,
            text="Open Details",
            bg="#007bff",
            fg="#ffffff",
            activebackground="#0056b3",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 9, "bold"),
            command=self.open_compare_details_window,
            padx=12,
            pady=6,
        ).pack(side=tk.RIGHT)

        self.compare_timer_active = False
        self.compare_timer_start = 0
        self.compare_waiting_for_removal = False
        self.compare_touch_counter = 0
        self.compare_trial_stopped = False
        self.compare_algos = {}
        self.compare_detail_log = []
        self.compare_details_window = None
        self.txt_compare_details = None
        self.reset_compare_trial()

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
        
        self.btn_refresh_sensors = tk.Button(src_frame, text="Refresh Camera", bg="#e9ecef", fg="#495057", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, command=self.refresh_camera)
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
        
        # Feature Toggles UI
        tk.Checkbutton(tab_setup, text="Enable Raw Feed", variable=self.enable_raw_var, command=self.rebuild_dashboard_grid, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=2)
        tk.Checkbutton(tab_setup, text="Enable Contact Heatmap", variable=self.enable_heatmap_var, command=self.rebuild_dashboard_grid, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=2)
        tk.Checkbutton(tab_setup, text="Enable Optical Flow", variable=self.enable_flow_var, command=self.rebuild_dashboard_grid, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=2)
        tk.Checkbutton(tab_setup, text="Enable 3D Reconstruction", variable=self.enable_reconstruction_var, command=self.rebuild_dashboard_grid, bg="#ffffff", fg="#212529", selectcolor="#ffffff", activebackground="#ffffff", activeforeground="#212529").pack(anchor=tk.W, pady=2)

        # Grid Layout Layout
        layout_frame = tk.Frame(tab_setup, bg="#ffffff")
        layout_frame.pack(fill=tk.X, pady=(5, 10))
        tk.Label(layout_frame, text="Grid Layout Columns:", bg="#ffffff", fg="#495057", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.cb_layout_cols = ttk.Combobox(layout_frame, textvariable=self.layout_cols_var, values=["Auto", "1", "2", "3", "4"], state="readonly", width=8)
        self.cb_layout_cols.pack(side=tk.LEFT, padx=5)
        self.cb_layout_cols.bind("<<ComboboxSelected>>", lambda e: self.rebuild_dashboard_grid())

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

        tk.Button(
            settings_footer,
            text="Close",
            bg="#e9ecef",
            fg="#495057",
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            bd=0,
            command=self.hide_settings_window,
            padx=18,
            pady=7,
        ).pack(side=tk.RIGHT, padx=(8, 0))
        self.btn_save_config = tk.Button(
            settings_footer,
            text="Save Settings",
            bg="#007bff",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            bd=0,
            command=self.save_config,
            padx=18,
            pady=7,
        )
        self.btn_save_config.pack(side=tk.RIGHT)
        
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

        self.show_main_view("predict")
        
        # Static status label below notebook, static at bottom
        self.status_lbl = tk.Label(sidebar_container, textvariable=self.status_var, font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#007bff", pady=5, bd=1, relief=tk.SUNKEN)
        self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
        
        self.fps_lbl = tk.Label(sidebar_container, text="FPS: 0.0", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#28a745", pady=5, bd=1, relief=tk.SUNKEN)
        self.fps_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
        
        # --- RIGHT AREA: 2X2 DASHBOARD GRID ---
        self.grid_frame = tk.Frame(main_frame, bg="#f8f9fa")
        self.grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure Grid Weights
        self.grid_frame.rowconfigure(0, weight=1)
        self.grid_frame.rowconfigure(1, weight=1)
        self.grid_frame.columnconfigure(0, weight=1)
        self.grid_frame.columnconfigure(1, weight=1)
        
        # 1. Raw Stream Canvas
        self.p1 = tk.LabelFrame(self.grid_frame, text="Raw Feed", bg="#ffffff", fg="#007bff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        self.lbl_raw = tk.Label(self.p1, bg="#f8f9fa")
        self.lbl_raw.pack(fill=tk.BOTH, expand=True)
        
        # 2. Difference Map Canvas
        self.p2 = tk.LabelFrame(self.grid_frame, text="Difference / Contact Heatmap", bg="#ffffff", fg="#007bff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        self.lbl_diff = tk.Label(self.p2, bg="#f8f9fa")
        self.lbl_diff.pack(fill=tk.BOTH, expand=True)
        
        # 3. Deformation Vectors Canvas
        self.p3 = tk.LabelFrame(self.grid_frame, text="Deformation Field Vectors (Optical Flow)", bg="#ffffff", fg="#007bff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        self.lbl_vectors = tk.Label(self.p3, bg="#f8f9fa")
        self.lbl_vectors.pack(fill=tk.BOTH, expand=True)
        
        # 4. 3D Mesh / Reconstruction (Matplotlib)
        self.p4 = tk.LabelFrame(self.grid_frame, text="3D Height Reconstruction", bg="#ffffff", fg="#007bff", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        
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
        
        # Call once to initialize grid layout based on default toggles
        self.rebuild_dashboard_grid()

    def rebuild_dashboard_grid(self):
        # 1. Hide all panels
        self.p1.grid_forget()
        self.p2.grid_forget()
        self.p3.grid_forget()
        self.p4.grid_forget()
        
        # 2. Reset weights
        self.grid_frame.rowconfigure(0, weight=0)
        self.grid_frame.rowconfigure(1, weight=0)
        self.grid_frame.columnconfigure(0, weight=0)
        self.grid_frame.columnconfigure(1, weight=0)
        
        # 3. Determine active panels
        active = []
        if self.enable_raw_var.get(): active.append(self.p1)
        if self.enable_heatmap_var.get(): active.append(self.p2)
        if self.enable_flow_var.get(): active.append(self.p3)
        if self.enable_reconstruction_var.get(): active.append(self.p4)
        
        if not active:
            return
            
        n = len(active)
        layout_mode = self.layout_cols_var.get()
        
        if layout_mode == "Auto":
            if n == 1:
                active[0].grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
                self.grid_frame.rowconfigure(0, weight=1)
                self.grid_frame.columnconfigure(0, weight=1)
            elif n == 2:
                active[0].grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
                active[1].grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
                self.grid_frame.rowconfigure(0, weight=1)
                self.grid_frame.columnconfigure(0, weight=1)
                self.grid_frame.columnconfigure(1, weight=1)
            elif n == 3:
                # 2 on top, 1 on bottom left
                active[0].grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
                active[1].grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
                active[2].grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
                self.grid_frame.rowconfigure(0, weight=1)
                self.grid_frame.rowconfigure(1, weight=1)
                self.grid_frame.columnconfigure(0, weight=1)
                self.grid_frame.columnconfigure(1, weight=1)
            elif n == 4:
                active[0].grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
                active[1].grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
                active[2].grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
                active[3].grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
                self.grid_frame.rowconfigure(0, weight=1)
                self.grid_frame.rowconfigure(1, weight=1)
                self.grid_frame.columnconfigure(0, weight=1)
                self.grid_frame.columnconfigure(1, weight=1)
        else:
            try:
                cols = int(layout_mode)
            except ValueError:
                cols = 2
            for i, panel in enumerate(active):
                r = i // cols
                c = i % cols
                panel.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
                self.grid_frame.rowconfigure(r, weight=1)
                self.grid_frame.columnconfigure(c, weight=1)
        
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
            "enable_raw": self.enable_raw_var.get(),
            "enable_heatmap": self.enable_heatmap_var.get(),
            "enable_flow": self.enable_flow_var.get(),
            "enable_reconstruction": self.enable_reconstruction_var.get(),
            "layout_cols": self.layout_cols_var.get(),
            "save_raw": self.save_raw_var.get(),
            "save_contact": self.save_contact_var.get(),
            "save_mask": self.save_mask_var.get(),
            "save_flow": self.save_flow_var.get(),
            "save_height_3d": self.save_height_3d_var.get(),
            "save_height_2d": self.save_height_2d_var.get(),
            "frame_scale": self.frame_scale_var.get(),
            "capture_mode": getattr(self, "capture_mode_var", tk.StringVar(value="Image")).get(),
            "auto_capture_threshold": getattr(self, "auto_capture_threshold", tk.IntVar(value=500)).get(),
            "prediction_threshold": self.prediction_threshold.get(),
            "prediction_timer_seconds": self.prediction_timer_seconds.get(),
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
        
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
        try:
            global_settings = {}
            if os.path.isfile(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded_settings = json.load(f)
                if isinstance(loaded_settings, dict):
                    global_settings.update(loaded_settings)
            global_settings["source"] = source_name
            global_settings["algorithm_configs"] = {
                key: variable.get().strip()
                for key, variable in self.algorithm_config_vars.items()
                if variable.get().strip()
            }
            with open(config_path, "w") as f:
                json.dump(global_settings, f, indent=4)
            self.reload_shape_algorithms()
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
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path, "r") as f:
                settings = json.load(f)
            
            if "source" in settings:
                self.source_var.set(str(settings["source"]))
            algorithm_configs = settings.get("algorithm_configs", {})
            if isinstance(algorithm_configs, dict):
                for key, variable in getattr(self, "algorithm_config_vars", {}).items():
                    value = algorithm_configs.get(key, "")
                    variable.set(str(value) if value else "")
                self.reload_shape_algorithms()
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

            set_val(self.enable_raw_var, "enable_raw", bool)
            set_val(self.enable_heatmap_var, "enable_heatmap", bool)
            set_val(self.enable_flow_var, "enable_flow", bool)
            set_val(self.enable_reconstruction_var, "enable_reconstruction", bool)
            set_val(self.layout_cols_var, "layout_cols", str)
            set_val(self.save_raw_var, "save_raw", bool)
            set_val(self.save_contact_var, "save_contact", bool)
            set_val(self.save_mask_var, "save_mask", bool)
            set_val(self.save_flow_var, "save_flow", bool)
            set_val(self.save_height_3d_var, "save_height_3d", bool)
            set_val(self.save_height_2d_var, "save_height_2d", bool)
            set_val(self.frame_scale_var, "frame_scale", float)
            set_val(getattr(self, "capture_mode_var", tk.StringVar(value="Image")), "capture_mode", str)
            set_val(getattr(self, "auto_capture_threshold", tk.IntVar(value=500)), "auto_capture_threshold", int)
            set_val(self.prediction_threshold, "prediction_threshold", int)
            set_val(self.prediction_timer_seconds, "prediction_timer_seconds", float)
            
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
            self.rebuild_dashboard_grid()
            
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
    
    def log_message(self, message):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        def append_log():
            self.txt_logs.config(state=tk.NORMAL)
            self.txt_logs.insert(tk.END, f"[{timestamp}] {message}\n")
            self.txt_logs.see(tk.END)
            self.txt_logs.config(state=tk.DISABLED)
        self.root.after(0, append_log)

    def update_treeview(self, probabilities):
        for item in self.tree_probs.get_children():
            self.tree_probs.delete(item)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            self.tree_probs.insert("", tk.END, values=(class_name, f"{prob:.1f}%"))

    def create_shape_algorithm(self, spec):
        variable = getattr(self, "algorithm_config_vars", {}).get(spec.key)
        config_path = variable.get().strip() if variable is not None else ""
        return spec.create(
            model_path=self.model_var.get(),
            config_path=config_path,
        )

    def reload_shape_algorithms(self):
        if hasattr(self, "shape_algo_var"):
            self.on_shape_algo_selected()
        if hasattr(self, "compare_algorithm_vars"):
            self.reset_compare_trial()

    def scan_optimization_run(self):
        optimize_directory = os.path.join(_root_dir, "optimize")
        selected = filedialog.askdirectory(
            initialdir=(
                optimize_directory
                if os.path.isdir(optimize_directory)
                else _root_dir
            ),
            title="Select Optimization Run or Model Folder",
        )
        if not selected:
            return

        selected_path = pathlib.Path(selected).resolve()
        if (selected_path / "run_manifest.json").is_file():
            run_path = selected_path
        else:
            manifests = list(selected_path.rglob("run_manifest.json"))
            if manifests:
                latest_manifest = max(
                    manifests,
                    key=lambda path: path.stat().st_mtime,
                )
                run_path = latest_manifest.parent
            else:
                run_path = selected_path

        specs_by_key = {spec.key: spec for spec in self.algorithm_specs}
        detected = {}
        for config_path in run_path.rglob("optimized_config.json"):
            key = config_path.parent.name
            if key in specs_by_key:
                current = detected.get(key)
                if current is None or config_path.stat().st_mtime > current.stat().st_mtime:
                    detected[key] = config_path.resolve()

        if not detected:
            messagebox.showwarning(
                "Optimization Run",
                f"No optimized_config.json files were found under:\n{run_path}",
            )
            return

        valid = {}
        invalid = {}
        for key, config_path in detected.items():
            spec = specs_by_key[key]
            try:
                algorithm = spec.create(config_path=str(config_path))
                self.validate_algorithm_features(algorithm)
                valid[key] = config_path
            except Exception as exc:
                invalid[key] = str(exc)

        if not valid:
            details = "\n".join(
                f"{specs_by_key[key].display_name}: {error}"
                for key, error in invalid.items()
            )
            messagebox.showerror(
                "Invalid Optimization Run",
                f"No usable configurations were found.\n\n{details}",
            )
            return

        for key, variable in self.algorithm_config_vars.items():
            variable.set(str(valid[key]) if key in valid else "")
        self.optimization_run_var.set(str(run_path))
        self.reload_shape_algorithms()

        loaded_names = [specs_by_key[key].display_name for key in valid]
        missing_names = [
            spec.display_name
            for spec in self.algorithm_specs
            if spec.key not in detected
        ]
        summary = "Loaded: " + ", ".join(loaded_names)
        if missing_names:
            summary += "\nAutomatic fallback: " + ", ".join(missing_names)
        if invalid:
            summary += "\nInvalid: " + ", ".join(
                specs_by_key[key].display_name for key in invalid
            )
        self.status_var.set(f"Status: Loaded Optimization {run_path.name}")
        messagebox.showinfo("Optimization Run Loaded", summary)

    def browse_algorithm_config(self, spec):
        variable = self.algorithm_config_vars[spec.key]
        current = variable.get().strip()
        default_directory = os.path.join(_root_dir, "algorithms", spec.key)
        initial_directory = (
            os.path.dirname(current)
            if current
            else default_directory
        )
        file_path = filedialog.askopenfilename(
            initialdir=initial_directory,
            title=f"Select {spec.display_name} Configuration",
            filetypes=(("JSON Configuration", "*.json"), ("All Files", "*.*")),
        )
        if not file_path:
            return
        try:
            algorithm = spec.create(config_path=file_path)
            self.validate_algorithm_features(algorithm)
        except Exception as exc:
            messagebox.showerror(
                "Invalid Algorithm Configuration",
                f"Could not load {spec.display_name}:\n\n{exc}",
            )
            return
        variable.set(os.path.abspath(file_path))
        self.reload_shape_algorithms()
        self.status_var.set(f"Status: Loaded {spec.display_name} Config")

    def clear_algorithm_config(self, spec):
        self.algorithm_config_vars[spec.key].set("")
        self.reload_shape_algorithms()
        self.status_var.set(f"Status: {spec.display_name} Config Automatic")

    def browse_weights(self):
        models_directory = os.path.join(_root_dir, "models")
        file_path = filedialog.askopenfilename(
            initialdir=models_directory if os.path.isdir(models_directory) else _root_dir,
            title="Select Model Weights",
            filetypes=(("PyTorch Model", "*.pth"), ("All Files", "*.*"))
        )
        if file_path:
            self.model_var.set(file_path)
            self.on_model_selected()

    def on_model_selected(self, event=None):
        arch = getattr(self, "arch_var", tk.StringVar(value="ResNet-18")).get()
        weights_path = self.model_var.get()
        
        if not weights_path or not os.path.exists(weights_path):
            self.update_treeview({})
            self.current_model = None
            self.class_names = []
            return
            
        try:
            # Derive labels path
            base, _ = os.path.splitext(weights_path)
            labels_path = base + ".txt"
            
            if os.path.exists(labels_path):
                with open(labels_path, "r") as f:
                    self.class_names = [line.strip() for line in f.readlines() if line.strip()]
            else:
                self.class_names = []
                self.log_message(f"Warning: Labels file not found at {labels_path}")
                
            num_classes = max(len(self.class_names), 1)
            
            if arch == "ResNet-18":
                import torch
                import torch.nn as nn
                from torchvision import models
                import torchvision.transforms as T
                from tools.model_calibration import load_encoder_temperature
                
                model = models.resnet18(weights=None)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
                
                state = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state)
                model.eval()
                
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(self.device)
                
                self.current_model = model
                self.encoder_temperature, calibration_path = load_encoder_temperature(
                    weights_path
                )
                self.encoder_calibrated = calibration_path.is_file()
                if calibration_path.is_file():
                    self.log_message(
                        f"Loaded encoder calibration: T={self.encoder_temperature:.4f} "
                        f"from {calibration_path}"
                    )
                else:
                    self.log_message(
                        f"No encoder calibration sidecar found at {calibration_path}; using T=1.0"
                    )
                
                self.val_transform = T.Compose([
                    T.Resize((256, 256)),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
            if self.class_names:
                self.update_treeview({c: 0.0 for c in self.class_names})
            else:
                self.update_treeview({f"Class {i}": 0.0 for i in range(num_classes)})
                
            self.log_message(f"Loaded {arch} from {weights_path}")
            self.on_shape_algo_selected()
            self.reset_compare_trial()
            
        except Exception as e:
            self.log_message(f"Error loading model: {e}")
            self.current_model = None
            self.class_names = []

    def on_shape_algo_selected(self, event=None):
        try:
            self.shape_algo_spec = get_algorithm_by_display_name(self.shape_algo_var.get())
            self.shape_algo = self.create_shape_algorithm(self.shape_algo_spec)
            self.validate_algorithm_features(self.shape_algo)
            config_path = getattr(self.shape_algo, "config_path", None)
            if config_path:
                self.log_message(f"Loaded {self.shape_algo_spec.display_name} config: {config_path}")
        except Exception as exc:
            self.shape_algo_spec = None
            self.shape_algo = None
            self.log_message(f"Failed to load shape algorithm: {exc}")
        self.reset_multi_trial()

    def validate_algorithm_features(self, algorithm):
        configured_features = getattr(algorithm, "features", None)
        model_features = getattr(self, "class_names", None)
        if not configured_features or not model_features:
            return

        aliases = getattr(algorithm, "aliases", {})
        normalized_model_features = {
            aliases.get(str(label).strip().lower(), str(label).strip().lower())
            for label in model_features
        }
        configured_features = set(configured_features)
        missing_from_model = sorted(configured_features - normalized_model_features)
        missing_from_config = sorted(normalized_model_features - configured_features)
        if missing_from_model or missing_from_config:
            details = []
            if missing_from_model:
                details.append("not produced by model: " + ", ".join(missing_from_model))
            if missing_from_config:
                details.append("not configured: " + ", ".join(missing_from_config))
            raise ValueError("Feature-label mismatch (" + "; ".join(details) + ")")

    def update_global_shape_display(self, result):
        if not result:
            return

        if isinstance(result, dict):
            best_shape = max(result, key=result.get)
            best_prob = result[best_shape] * 100
            self.global_shape_var.set(f"Global Shape: {best_shape.upper()} ({best_prob:.1f}%)")
            return

        belief = getattr(result, "belief", None)
        prediction = getattr(result, "prediction", None)
        if belief and prediction:
            best_prob = belief.get(prediction, 0.0) * 100
            if getattr(result, "is_uncertain", False):
                self.global_shape_var.set(
                    f"Global Shape: UNCERTAIN (best {prediction.upper()}, {best_prob:.1f}%)"
                )
            else:
                stop_text = " | ACCEPT" if getattr(result, "should_stop", False) else ""
                self.global_shape_var.set(
                    f"Global Shape: {prediction.upper()} ({best_prob:.1f}%){stop_text}"
                )

    def summarize_shape_result(self, result):
        if not result:
            return "unknown", 0.0, {}, False, False, ""

        if isinstance(result, dict):
            probabilities = {str(name): float(prob) for name, prob in result.items()}
            prediction = max(probabilities, key=probabilities.get)
            return (
                prediction,
                probabilities[prediction] * 100.0,
                probabilities,
                False,
                False,
                "",
            )

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

    def export_multi_touch_csv(self):
        if not getattr(self, "multi_touch_history", None):
            messagebox.showinfo("Export Multi Touch", "No multi-touch predictions to export.")
            return

        import datetime

        default_name = f"multi_touch_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = filedialog.asksaveasfilename(
            title="Export Multi-Touch Predictions",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
        )
        if not file_path:
            return

        fieldnames = [
            "touch",
            "feature",
            "global_shape_confidence",
        ]

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.multi_touch_history:
                    writer.writerow({
                        "touch": record["touch"],
                        "feature": record["top_local_feature"],
                        "global_shape_confidence": (
                            f"{record['predicted_shape']} "
                            f"({record['shape_confidence_percent']:.1f}%)"
                        ),
                    })
        except Exception as exc:
            messagebox.showerror("Export Multi Touch", f"Failed to export CSV:\n\n{exc}")
            return

        self.status_var.set(f"Status: Exported Multi Touch CSV")
        self.log_message(f"Exported multi-touch predictions to {file_path}")
        messagebox.showinfo("Export Multi Touch", f"Saved CSV:\n{file_path}")

    def reset_multi_trial(self):
        for item in self.tree_multi.get_children():
            self.tree_multi.delete(item)
        self.multi_touch_counter = 0
        self.multi_trial_stopped = False
        self.multi_touch_history = []
        if hasattr(self, 'shape_algo') and hasattr(self.shape_algo, 'reset'):
            self.shape_algo.reset()
        self.global_shape_var.set("Global Shape: Unknown (0%)")
        self.multi_status_var.set("Waiting for Object...")

    def build_compare_algorithms(self):
        algorithms = {}
        variables = getattr(self, "compare_algorithm_vars", {})
        for spec in getattr(self, "algorithm_specs", ()):
            enabled = variables.get(spec.key)
            if enabled is None or not enabled.get():
                continue
            try:
                algorithm = self.create_shape_algorithm(spec)
                self.validate_algorithm_features(algorithm)
                algorithms[spec.key] = (spec, algorithm)
            except Exception as exc:
                self.log_message(f"Failed to load {spec.display_name}: {exc}")
        return algorithms

    def open_compare_details_window(self):
        if (
            self.compare_details_window is not None
            and self.compare_details_window.winfo_exists()
        ):
            self.compare_details_window.deiconify()
            self.compare_details_window.lift()
            self.compare_details_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        window.title("Algorithm Comparison Details")
        window.geometry("760x620")
        window.minsize(560, 420)
        window.configure(bg="#f8f9fa")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", window.withdraw)
        self.compare_details_window = window

        header = tk.Frame(window, bg="#ffffff", padx=16, pady=12)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="Comparison Details",
            bg="#ffffff",
            fg="#212529",
            font=("Segoe UI", 15, "bold"),
        ).pack(side=tk.LEFT)
        tk.Button(
            header,
            text="Close",
            bg="#e9ecef",
            fg="#495057",
            relief=tk.FLAT,
            bd=0,
            command=window.withdraw,
            padx=14,
            pady=6,
        ).pack(side=tk.RIGHT)

        text_frame = tk.Frame(window, bg="#f8f9fa")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        vertical = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        horizontal = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
        self.txt_compare_details = tk.Text(
            text_frame,
            bg="#ffffff",
            fg="#212529",
            font=("Consolas", 10),
            state=tk.DISABLED,
            wrap=tk.NONE,
            padx=12,
            pady=12,
            yscrollcommand=vertical.set,
            xscrollcommand=horizontal.set,
        )
        vertical.config(command=self.txt_compare_details.yview)
        horizontal.config(command=self.txt_compare_details.xview)
        self.txt_compare_details.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.render_compare_details()

    def render_compare_details(self):
        if (
            self.txt_compare_details is None
            or not self.txt_compare_details.winfo_exists()
        ):
            return
        content = "\n\n".join(self.compare_detail_log)
        if not content:
            content = "No comparison details recorded."
        self.txt_compare_details.config(state=tk.NORMAL)
        self.txt_compare_details.delete("1.0", tk.END)
        self.txt_compare_details.insert(tk.END, content)
        self.txt_compare_details.config(state=tk.DISABLED)
        self.txt_compare_details.see(tk.END)

    def reset_compare_trial(self):
        if hasattr(self, "tree_compare"):
            for item in self.tree_compare.get_children():
                self.tree_compare.delete(item)
        self.compare_touch_counter = 0
        self.compare_timer_active = False
        self.compare_waiting_for_removal = False
        self.compare_trial_stopped = False
        self.compare_algos = self.build_compare_algorithms() if hasattr(self, "compare_algorithm_vars") else {}
        if hasattr(self, "compare_status_var"):
            self.compare_status_var.set("Waiting for Object...")
        if hasattr(self, "compare_result_summary_var"):
            self.compare_result_summary_var.set("Waiting for first touch")
        if hasattr(self, "compare_detail_log"):
            self.compare_detail_log = []
            self.compare_details_summary_var.set("No details recorded")
            self.render_compare_details()

    def format_probability_block(self, title, probs):
        lines = [f"{title}:"]
        for name, value in sorted(probs.items(), key=lambda x: -x[1]):
            lines.append(f"  {name:<16} {value * 100:6.2f}%")
        return "\n".join(lines)

    def append_compare_details(self, text):
        self.compare_detail_log.append(text)
        touch_count = len(self.compare_detail_log)
        suffix = "touch" if touch_count == 1 else "touches"
        self.compare_details_summary_var.set(f"Details available for {touch_count} {suffix}")
        self.render_compare_details()

    def update_compare_results_display(self, local_results, algorithm_results):
        for item in self.tree_compare.get_children():
            self.tree_compare.delete(item)

        detail_lines = []
        displayed_predictions = []
        top_feature = max(local_results, key=local_results.get)
        detail_lines.append(f"Touch {self.compare_touch_counter}: {top_feature.upper()} ({local_results[top_feature]:.1f}%)")
        detail_lines.append("")
        detail_lines.append("Local feature probabilities:")
        for name, value in sorted(local_results.items(), key=lambda x: -x[1]):
            detail_lines.append(f"  {name:<18} {value:6.2f}%")

        for algo_name, result in algorithm_results.items():
            if isinstance(result, dict):
                prediction = max(result, key=result.get)
                confidence = result[prediction]
                probs = result
                row_tag = "active"
                detail_lines.append("")
                detail_lines.append(self.format_probability_block(algo_name, probs))
            else:
                prediction = getattr(result, "prediction", "unknown")
                belief = getattr(result, "belief", {})
                confidence = belief.get(prediction, 0.0)
                is_uncertain = getattr(result, "is_uncertain", False)
                displayed_prediction = "UNCERTAIN" if is_uncertain else prediction.upper()
                detail_lines.append("")
                detail_lines.append(self.format_probability_block(algo_name, belief))
                detail_lines.append(f"  entropy:           {getattr(result, 'entropy', 0.0):.4f}")
                detail_lines.append(f"  touches:           {getattr(result, 'touch_count', 0)}")
                coverage = getattr(result, "feature_coverage", {})
                if coverage:
                    detail_lines.append("  feature coverage:")
                    for fname, fval in sorted(coverage.items(), key=lambda x: -x[1]):
                        if fval > 0.001:
                            detail_lines.append(f"    {fname:<16} {fval:6.3f}")
                stopping_reason = getattr(result, "stopping_reason", None)
                if stopping_reason == "confidence":
                    row_tag = "accepted"
                    detail_lines.append("  decision: ACCEPTED (confidence criteria met)")
                elif stopping_reason == "max_touches":
                    row_tag = "uncertain"
                    detail_lines.append(
                        f"  decision: UNCERTAIN (best class: {prediction.upper()})"
                    )
                else:
                    row_tag = "active"
            if isinstance(result, dict):
                displayed_prediction = prediction.upper()

            displayed_predictions.append(displayed_prediction)

            self.tree_compare.insert(
                "",
                tk.END,
                values=(
                    algo_name,
                    displayed_prediction,
                    f"{confidence * 100:.1f}%",
                ),
                tags=(row_tag,),
            )

        unique_predictions = set(displayed_predictions)
        if not displayed_predictions:
            self.compare_result_summary_var.set("No algorithm results")
        elif len(displayed_predictions) == 1:
            self.compare_result_summary_var.set(
                f"Prediction: {displayed_predictions[0]}"
            )
        elif len(unique_predictions) == 1:
            self.compare_result_summary_var.set(
                f"Agreement: {displayed_predictions[0]}"
            )
        else:
            self.compare_result_summary_var.set("Algorithms disagree")

        self.append_compare_details("\n".join(detail_lines))

    def run_comparison_prediction(self, frame):
        if getattr(self, "current_model", None) is None:
            self.root.after(0, lambda: self.compare_status_var.set("Error: No Model Selected!"))
            self.root.after(2000, lambda: self.compare_status_var.set("Waiting for Object..."))
            return

        try:
            import cv2
            from PIL import Image
            import torch

            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            img = Image.fromarray(frame_rgb)
            tensor = self.val_transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.current_model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]

            local_results = {}
            for i, prob in enumerate(probs):
                name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                local_results[name] = prob.item() * 100

            if not local_results:
                return

            def append_compare_result():
                if not self.compare_algos:
                    self.compare_algos = self.build_compare_algorithms()
                if not self.compare_algos:
                    self.compare_status_var.set("Select at least one algorithm.")
                    return

                self.compare_touch_counter += 1
                top_feature = max(local_results, key=local_results.get)
                self.compare_status_var.set(f"Touch {self.compare_touch_counter}: {top_feature.upper()}")

                algorithm_results = {}
                feature_probs = {name: prob / 100.0 for name, prob in local_results.items()}
                for spec, algorithm in self.compare_algos.values():
                    algorithm_results[spec.display_name] = spec.update(
                        algorithm,
                        feature_probabilities=feature_probs,
                        top_feature=top_feature,
                    )

                self.update_compare_results_display(local_results, algorithm_results)
                stopping_results = [
                    result
                    for result in algorithm_results.values()
                    if getattr(result, "should_stop", False)
                ]
                if stopping_results:
                    self.compare_trial_stopped = True
                    if any(
                        getattr(result, "is_uncertain", False)
                        for result in stopping_results
                    ):
                        self.compare_status_var.set("Trial stopped: uncertain")
                    else:
                        self.compare_status_var.set("Trial stopped: accepted")

            self.root.after(0, append_compare_result)
        except Exception as e:
            self.log_message(f"Comparison Prediction Error: {e}")
            self.root.after(0, lambda: self.compare_status_var.set("Prediction Error!"))

    def run_prediction(self, frame):
        if getattr(self, "current_model", None) is None:
            self.root.after(0, lambda: self.pred_result_var.set("No Model Selected"))
            return
            
        try:
            import cv2
            from PIL import Image
            import torch
            from tools.model_calibration import apply_temperature
            
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
                
            img = Image.fromarray(frame_rgb)
            tensor = self.val_transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = apply_temperature(
                    self.current_model(tensor),
                    getattr(self, "encoder_temperature", 1.0),
                )
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                
            results = {}
            for i, prob in enumerate(probs):
                name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                results[name] = prob.item() * 100
                
            if results:
                top_class = max(results, key=results.get)
                self.root.after(0, lambda: self.pred_result_var.set(top_class.upper()))
                self.root.after(0, lambda: self.update_treeview(results))
        except Exception as e:
            self.log_message(f"Prediction Error: {e}")
            self.root.after(0, lambda: self.pred_result_var.set("Error - Check Logs"))

    def run_multi_prediction(self, frame):
        if getattr(self, "current_model", None) is None:
            self.root.after(0, lambda: self.multi_status_var.set("Error: No Model Selected!"))
            self.root.after(2000, lambda: self.multi_status_var.set("Waiting for Object..."))
            return
            
        try:
            import cv2
            from PIL import Image
            import torch
            import datetime
            from tools.model_calibration import apply_temperature
            
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
                
            img = Image.fromarray(frame_rgb)
            tensor = self.val_transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = apply_temperature(
                    self.current_model(tensor),
                    getattr(self, "encoder_temperature", 1.0),
                )
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                
            results = {}
            for i, prob in enumerate(probs):
                name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                results[name] = prob.item() * 100
                
            if results:
                top_class = max(results, key=results.get)
                timestamp = datetime.datetime.now().isoformat(timespec="seconds")
                
                def append_multi_result():
                    self.multi_touch_counter += 1
                    feature_probs = {name: prob / 100.0 for name, prob in results.items()}
                    updated_result = None
                    
                    if (
                        getattr(self, "shape_algo", None) is not None
                        and getattr(self, "shape_algo_spec", None) is not None
                    ):
                        updated_result = self.shape_algo_spec.update(
                            self.shape_algo,
                            feature_probabilities=feature_probs,
                            top_feature=top_class,
                        )
                        self.update_global_shape_display(updated_result)

                    (
                        predicted_shape,
                        shape_confidence,
                        shape_probabilities,
                        should_stop,
                        is_uncertain,
                        stopping_reason,
                    ) = self.summarize_shape_result(updated_result)
                    self.tree_multi.insert(
                        "",
                        tk.END,
                        values=(
                            self.multi_touch_counter,
                            top_class.upper(),
                            f"{predicted_shape.upper()} ({shape_confidence:.1f}%)",
                        ),
                    )
                    self.tree_multi.yview_moveto(1) # Auto scroll to bottom

                    self.multi_touch_history.append(
                        {
                            "touch": self.multi_touch_counter,
                            "timestamp": timestamp,
                            "algorithm": (
                                self.shape_algo_spec.display_name
                                if getattr(self, "shape_algo_spec", None) is not None
                                else ""
                            ),
                            "top_local_feature": top_class,
                            "feature_probabilities": feature_probs,
                            "predicted_shape": predicted_shape,
                            "shape_confidence_percent": shape_confidence,
                            "shape_probabilities": shape_probabilities,
                            "should_stop": should_stop,
                            "is_uncertain": is_uncertain,
                            "stopping_reason": stopping_reason,
                        }
                    )

                    self.multi_status_var.set(
                        f"Touch {self.multi_touch_counter}: {predicted_shape.upper()} ({shape_confidence:.1f}%)"
                    )
                    if should_stop:
                        self.multi_trial_stopped = True
                        if is_uncertain:
                            self.multi_status_var.set("Trial stopped: uncertain")
                        else:
                            self.multi_status_var.set("Trial stopped: accepted")
                    
                self.root.after(0, append_multi_result)
        except Exception as e:
            self.log_message(f"Multi Prediction Error: {e}")
            self.root.after(0, lambda: self.multi_status_var.set("Prediction Error!"))

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
            curr_t = time.time()
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

            if curr_t - getattr(self, 'last_pred_time', 0) >= getattr(self, 'pred_interval', 0.5):
                self.last_pred_time = curr_t

                if pred_frame is not None:
                    try:
                        thresh = self.prediction_threshold.get()
                    except Exception:
                        thresh = 100
                        
                    if getattr(self, 'current_contact_area', 0) >= thresh:
                        # Copy frame for thread safety
                        pf_copy = pred_frame.copy()
                        # Run in background thread to avoid blocking video loop
                        threading.Thread(target=self.run_prediction, args=(pf_copy,), daemon=True).start()
                    else:
                        self.root.after(0, lambda: self.pred_result_var.set("NO OBJECT"))
                        def reset_probs():
                            for item in self.tree_probs.get_children():
                                vals = self.tree_probs.item(item, 'values')
                                if vals:
                                    self.tree_probs.item(item, values=(vals[0], "0.0%"))
                        self.root.after(0, reset_probs)


            # --- MULTI TOUCH PREDICTION ---
            try:
                if self.active_main_view == "multi":
                    multi_thresh = self.prediction_threshold.get()
                    contact_area = getattr(self, 'current_contact_area', 0)
                    
                    if self.multi_trial_stopped:
                        pass
                    elif contact_area >= multi_thresh:
                        if self.multi_waiting_for_removal:
                            pass
                        elif not self.multi_timer_active:
                            self.multi_timer_active = True
                            self.multi_timer_start = curr_t
                            self.root.after(0, lambda: self.multi_status_var.set("Detecting..."))
                        else:
                            elapsed = curr_t - self.multi_timer_start
                            duration = self.prediction_timer_seconds.get()
                            remaining = duration - elapsed
                            
                            if remaining > 0:
                                self.root.after(0, lambda r=remaining: self.multi_status_var.set(f"Capturing in {r:.1f}s"))
                            else:
                                self.multi_timer_active = False
                                self.multi_waiting_for_removal = True
                                self.root.after(0, lambda: self.multi_status_var.set("Predicting..."))
                                
                                if pred_frame is not None:
                                    pf_copy = pred_frame.copy()
                                    threading.Thread(target=self.run_multi_prediction, args=(pf_copy,), daemon=True).start()
                    else:
                        if self.multi_timer_active:
                            self.multi_timer_active = False
                            self.root.after(0, lambda: self.multi_status_var.set("Waiting for Object..."))
                        elif self.multi_waiting_for_removal:
                            self.multi_waiting_for_removal = False
            except Exception:
                pass

            # --- SHAPE ALGORITHM COMPARISON ---
            try:
                if self.active_main_view == "compare":
                    compare_thresh = self.prediction_threshold.get()
                    contact_area = getattr(self, 'current_contact_area', 0)

                    if self.compare_trial_stopped:
                        pass
                    elif contact_area >= compare_thresh:
                        if self.compare_waiting_for_removal:
                            pass
                        elif not self.compare_timer_active:
                            self.compare_timer_active = True
                            self.compare_timer_start = curr_t
                            self.root.after(0, lambda: self.compare_status_var.set("Detecting..."))
                        else:
                            elapsed = curr_t - self.compare_timer_start
                            duration = self.prediction_timer_seconds.get()
                            remaining = duration - elapsed

                            if remaining > 0:
                                self.root.after(0, lambda r=remaining: self.compare_status_var.set(f"Capturing in {r:.1f}s"))
                            else:
                                self.compare_timer_active = False
                                self.compare_waiting_for_removal = True
                                self.root.after(0, lambda: self.compare_status_var.set("Predicting..."))

                                if pred_frame is not None:
                                    pf_copy = pred_frame.copy()
                                    threading.Thread(target=self.run_comparison_prediction, args=(pf_copy,), daemon=True).start()
                    else:
                        if self.compare_timer_active:
                            self.compare_timer_active = False
                            self.root.after(0, lambda: self.compare_status_var.set("Waiting for Object..."))
                        elif self.compare_waiting_for_removal:
                            self.compare_waiting_for_removal = False
            except Exception:
                pass


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
            if self.enable_raw_var.get():
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
            elif hasattr(self, 'Z_plot'):
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
            initial_dir = os.path.dirname(__file__)
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
