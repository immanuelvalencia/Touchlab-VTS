import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
from datetime import datetime
import threading
import time

class CameraCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Camera Calibration Tool")
        self.root.geometry("1100x750")
        self.root.configure(bg="#f8f9fa")
        
        # White/Light UI Styles matching annotate.py
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#f8f9fa", foreground="#212529", fieldbackground="#ffffff")
        self.style.configure("TLabel", background="#f8f9fa", foreground="#212529", font=("Segoe UI", 10))
        self.style.configure("TButton", background="#007bff", foreground="#ffffff", borderwidth=0, font=("Segoe UI", 10, "bold"))
        self.style.map("TButton", background=[("active", "#0056b3")])
        self.style.configure("TEntry", fieldbackground="#ffffff", foreground="#212529")
        self.style.configure("TLabelframe", background="#f8f9fa", foreground="#212529")
        self.style.configure("TLabelframe.Label", background="#f8f9fa", foreground="#212529", font=("Segoe UI", 10, "bold"))
        self.style.configure("TCombobox", fieldbackground="#ffffff", background="#e9ecef", foreground="#212529")
        self.style.configure("TCheckbutton", background="#f8f9fa", foreground="#212529", font=("Segoe UI", 10))
        
        # Variables
        self.camera_name = tk.StringVar(value="Camera 1")
        self.fov = tk.DoubleVar(value=90.0)
        self.board_rows = tk.IntVar(value=7)
        self.board_cols = tk.IntVar(value=9)
        self.square_size = tk.DoubleVar(value=0.025) # in meters
        self.alpha_var = tk.DoubleVar(value=0.0) # 0 = crop (flat), 1 = retain all pixels
        self.resolution_var = tk.StringVar(value="Auto Detect")
        
        # Video Capture State
        self.cap = None
        self.camera_index = tk.IntVar(value=0)
        self.is_running = True
        
        # Auto Capture State
        self.auto_capture_var = tk.BooleanVar(value=True)
        self.last_capture_time = 0
        self.capture_delay = 1.5 # Seconds between auto captures
        
        # Calibration State
        self.captured_images = []
        self.captured_corners = []
        self.calibration_results = None
        self.img_size = None
        self.new_camera_mtx = None
        self.roi = None
        
        self.setup_ui()
        
        # Start Video Thread
        self.start_camera()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        # Main Layout
        self.left_panel = ttk.Frame(self.root, width=320)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.right_panel = ttk.Frame(self.root)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Settings (Left Panel) ---
        settings_frame = ttk.LabelFrame(self.left_panel, text="Settings")
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Camera Index:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        idx_entry = ttk.Entry(settings_frame, textvariable=self.camera_index)
        idx_entry.grid(row=0, column=1, padx=5, pady=5)
        idx_entry.bind('<Return>', lambda e: self.change_camera_index())
        
        ttk.Label(settings_frame, text="Camera Name:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(settings_frame, textvariable=self.camera_name).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="FOV (degrees):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(settings_frame, textvariable=self.fov).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Checkerboard Rows:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(settings_frame, textvariable=self.board_rows).grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Checkerboard Cols:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(settings_frame, textvariable=self.board_cols).grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Square Size (m):").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(settings_frame, textvariable=self.square_size).grid(row=5, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Alpha (0=Flat, 1=Full):").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        alpha_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.alpha_var, command=self.update_alpha)
        alpha_scale.grid(row=6, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(settings_frame, text="Resolution:").grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
        self.res_cb = ttk.Combobox(settings_frame, textvariable=self.resolution_var, state="readonly", width=18)
        self.res_cb['values'] = ("Auto Detect", "Native/Max")
        self.res_cb.grid(row=7, column=1, padx=5, pady=5, sticky=tk.W)
        self.res_cb.bind('<<ComboboxSelected>>', lambda e: self.start_camera())
        
        # --- Actions (Left Panel) ---
        actions_frame = ttk.LabelFrame(self.left_panel, text="Calibration Actions")
        actions_frame.pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(actions_frame, text="Auto Detect & Capture", variable=self.auto_capture_var).pack(anchor=tk.W, padx=5, pady=5)
        ttk.Label(actions_frame, text="(Move the checkerboard around slowly)").pack(anchor=tk.W, padx=5)
        
        self.btn_capture = ttk.Button(actions_frame, text="Manual Capture", command=self.capture_frame)
        self.btn_capture.pack(fill=tk.X, padx=5, pady=5)
        
        self.lbl_image_count = ttk.Label(actions_frame, text="0 frames captured", font=("Segoe UI", 10, "bold"))
        self.lbl_image_count.pack(pady=5)
        
        self.btn_calibrate = ttk.Button(actions_frame, text="Calibrate Camera", command=self.calibrate_camera)
        self.btn_calibrate.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_clear = ttk.Button(actions_frame, text="Clear Frames", command=self.clear_frames)
        self.btn_clear.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_load_json = ttk.Button(actions_frame, text="Load Calibration JSON", command=self.load_json)
        self.btn_load_json.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_export_json = ttk.Button(actions_frame, text="Export JSON", command=self.export_json, state=tk.DISABLED)
        self.btn_export_json.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_export_report = ttk.Button(actions_frame, text="Export Markdown Report", command=self.export_report, state=tk.DISABLED)
        self.btn_export_report.pack(fill=tk.X, padx=5, pady=5)
        
        self.lbl_status = ttk.Label(self.left_panel, text="Status: Ready", foreground="#28a745", wraplength=280)
        self.lbl_status.pack(pady=10)
        
        # --- Frustum Calculator (Left Panel) ---
        frustum_frame = ttk.LabelFrame(self.left_panel, text="Frustum Calculator")
        frustum_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(frustum_frame, text="Distance Z (mm):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.calc_z = tk.DoubleVar(value=100.0)
        z_entry = ttk.Entry(frustum_frame, textvariable=self.calc_z, width=10)
        z_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        z_entry.bind('<KeyRelease>', self.update_frustum)
        
        self.lbl_calc_w = ttk.Label(frustum_frame, text="Width: - mm")
        self.lbl_calc_w.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        self.lbl_calc_h = ttk.Label(frustum_frame, text="Height: - mm")
        self.lbl_calc_h.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        self.lbl_calc_area = ttk.Label(frustum_frame, text="Area: - mm²")
        self.lbl_calc_area.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        # --- Preview (Right Panel) ---
        self.paned_window = ttk.PanedWindow(self.right_panel, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, pady=5)
        
        preview_frame = ttk.LabelFrame(self.paned_window, text="Live Camera Feed (Raw)")
        self.canvas = tk.Canvas(preview_frame, bg="#e9ecef")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.paned_window.add(preview_frame, weight=1)
        
        undistort_frame = ttk.LabelFrame(self.paned_window, text="Live Camera Feed (Undistorted)")
        self.undistort_canvas = tk.Canvas(undistort_frame, bg="#e9ecef")
        self.undistort_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.paned_window.add(undistort_frame, weight=1)
        
    def change_camera_index(self):
        if hasattr(self, 'res_cb_populated'):
            self.res_cb_populated = False
        self.start_camera()
        
    def get_supported_resolutions(self):
        test_resolutions = [
            (3840, 2160), (2560, 1440), (2048, 1536), (1920, 1080), 
            (1600, 1200), (1280, 1024), (1280, 720), (1024, 768), 
            (800, 600), (640, 480), (480, 360), (320, 240)
        ]
        supported = []
        for w, h in test_resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w == w and actual_h == h:
                ratio_val = w / h
                if abs(ratio_val - 16/9) < 0.05: ratio = "16:9"
                elif abs(ratio_val - 4/3) < 0.05: ratio = "4:3"
                elif abs(ratio_val - 5/4) < 0.05: ratio = "5:4"
                else: ratio = "Custom"
                res_str = f"{w}x{h} ({ratio})"
                if res_str not in supported:
                    supported.append(res_str)
        return supported

    def start_camera(self):
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.camera_index.get(), cv2.CAP_DSHOW)
        
        if not hasattr(self, 'res_cb_populated'):
            self.res_cb_populated = False
            
        if not self.res_cb_populated and self.cap.isOpened():
            self.lbl_status.config(text="Probing supported resolutions...", foreground="#007bff")
            self.root.update()
            supported = self.get_supported_resolutions()
            if supported:
                self.res_cb['values'] = ["Native/Max"] + supported
                if self.resolution_var.get() == "Auto Detect":
                    self.resolution_var.set(supported[0])
            else:
                self.res_cb['values'] = ["Native/Max", "Auto Detect"]
            self.res_cb_populated = True
        
        res_selection = self.resolution_var.get()
        if res_selection == "Native/Max" or res_selection == "Auto Detect":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        else:
            try:
                res_clean = res_selection.split(' ')[0]
                w, h = map(int, res_clean.split('x'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            except ValueError:
                pass
            
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            self.lbl_status.config(text="Status: Error opening camera", foreground="#dc3545")
            return
            
        self.lbl_status.config(text=f"Status: Camera started at {int(actual_w)}x{int(actual_h)}", foreground="#28a745")
            
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()
        
    def change_camera(self):
        self.start_camera()

    def video_loop(self):
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    display_frame = frame.copy()
                    
                    if self.auto_capture_var.get():
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        rows = self.board_rows.get()
                        cols = self.board_cols.get()
                        
                        # Find corners
                        ret_corners, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
                        
                        if ret_corners:
                            # Draw corners on display frame for feedback
                            cv2.drawChessboardCorners(display_frame, (cols, rows), corners, ret_corners)
                            
                            # Check if enough time has passed since last capture
                            current_time = time.time()
                            if current_time - self.last_capture_time > self.capture_delay:
                                # Subpixel refinement
                                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                                
                                # Only capture if it moved significantly from the last capture to avoid identical frames
                                should_capture = True
                                if self.captured_corners:
                                    last_corners = self.captured_corners[-1]
                                    diff = np.linalg.norm(corners2 - last_corners)
                                    if diff < 20.0: # threshold for movement
                                        should_capture = False
                                        
                                if should_capture:
                                    self.last_capture_time = current_time
                                    
                                    if self.img_size is None:
                                        self.img_size = gray.shape[::-1]
                                        
                                    self.captured_images.append(frame)
                                    self.captured_corners.append(corners2)
                                    
                                    # Update UI from background thread
                                    self.root.after(0, self.update_capture_ui)

                    # Send to canvas
                    self.root.after(0, self.update_canvas, display_frame)
                    
                    if self.calibration_results is not None and self.new_camera_mtx is not None:
                        mtx = np.array(self.calibration_results['camera_matrix'])
                        dist = np.array(self.calibration_results['distortion_coefficients'])
                        undistorted = cv2.undistort(frame, mtx, dist, None, self.new_camera_mtx)
                        self.root.after(0, self.update_undistort_canvas, undistorted)
            time.sleep(0.03) # ~30 fps cap
            
    def update_capture_ui(self):
        self.lbl_image_count.config(text=f"{len(self.captured_images)} frames captured")
        self.lbl_status.config(text=f"Status: Auto-captured frame!", foreground="#007bff")
        
        # Flash button
        self.btn_capture.config(text="✅ Auto Captured!")
        self.root.after(500, lambda: self.btn_capture.config(text="Manual Capture"))

    def update_canvas(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 10 and canvas_h > 10:
            pil_img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
            
        self.preview_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=self.preview_image)
        
    def update_undistort_canvas(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        canvas_w = self.undistort_canvas.winfo_width()
        canvas_h = self.undistort_canvas.winfo_height()
        
        if canvas_w > 10 and canvas_h > 10:
            pil_img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
            
        self.undistort_preview_image = ImageTk.PhotoImage(pil_img)
        self.undistort_canvas.delete("all")
        self.undistort_canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=self.undistort_preview_image)
        
    def capture_frame(self):
        # Manual capture logic
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Warning", "Camera is not active.")
            return
            
        ret, frame = self.cap.read()
        if not ret: return
        
        self.lbl_status.config(text="Processing manual frame...", foreground="#007bff")
        self.root.update()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows = self.board_rows.get()
        cols = self.board_cols.get()
        
        if self.img_size is None:
            self.img_size = gray.shape[::-1]
            
        ret_corners, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        
        if ret_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            self.captured_images.append(frame)
            self.captured_corners.append(corners2)
            
            self.lbl_image_count.config(text=f"{len(self.captured_images)} frames captured")
            self.lbl_status.config(text=f"Status: Manual frame captured!", foreground="#28a745")
            self.last_capture_time = time.time() # Reset auto-capture timer
        else:
            self.lbl_status.config(text=f"Status: Chessboard not fully visible.", foreground="#dc3545")

    def clear_frames(self):
        self.captured_images.clear()
        self.captured_corners.clear()
        self.calibration_results = None
        self.img_size = None
        self.new_camera_mtx = None
        self.roi = None
        self.alpha_var.set(0.0)
        self.lbl_image_count.config(text="0 frames captured")
        self.btn_export_json.config(state=tk.DISABLED)
        self.btn_export_report.config(state=tk.DISABLED)
        self.lbl_status.config(text="Status: Frames cleared.", foreground="#28a745")
        self.undistort_canvas.delete("all")
        self.lbl_calc_w.config(text="Width: - mm")
        self.lbl_calc_h.config(text="Height: - mm")
        self.lbl_calc_area.config(text="Area: - mm²")
        
    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                self.calibration_results = data
                self.img_size = tuple(data['image_size'])
                self.camera_name.set(data.get('camera_name', 'Loaded Camera'))
                self.fov.set(data.get('fov', 90.0))
                
                # compute new camera matrix based on current alpha
                self.update_alpha()
                
                self.btn_export_json.config(state=tk.NORMAL)
                self.btn_export_report.config(state=tk.NORMAL)
                self.lbl_status.config(text=f"Status: Loaded {os.path.basename(path)}", foreground="#28a745")
                self.update_frustum()
                
                messagebox.showinfo("Success", f"Calibration JSON loaded successfully.\nLive preview will now use this calibration.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load calibration:\n{str(e)}")

    def calibrate_camera(self):
        if len(self.captured_corners) < 3:
            messagebox.showwarning("Warning", f"Need at least 3 captured frames with detected corners. Currently have {len(self.captured_corners)}.")
            return
            
        self.lbl_status.config(text="Calibrating... please wait.", foreground="#007bff")
        self.root.update()
        
        try:
            rows = self.board_rows.get()
            cols = self.board_cols.get()
            square_size = self.square_size.get()
            
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            objp *= square_size
            
            objpoints = [objp] * len(self.captured_corners)
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, self.captured_corners, self.img_size, None, None)
            
            alpha = self.alpha_var.get()
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, self.img_size, alpha, self.img_size)
            self.new_camera_mtx = newcameramtx
            self.roi = roi
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.calibration_results = {
                "camera_name": self.camera_name.get(),
                "timestamp": timestamp_str,
                "fov": self.fov.get(),
                "image_size": [self.img_size[0], self.img_size[1]],
                "rms_error": ret,
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist(),
                "new_camera_matrix": newcameramtx.tolist(),
                "roi": roi,
                "valid_images_used": len(self.captured_images),
                "total_images": len(self.captured_images)
            }
            
            self.lbl_status.config(text=f"Status: Calibration complete! RMS Error: {ret:.4f}", foreground="#28a745")
            self.btn_export_json.config(state=tk.NORMAL)
            self.btn_export_report.config(state=tk.NORMAL)
            
            # Auto export
            self.export_json()
            self.export_report()
            self.update_frustum()
            
            cam_name_safe = self.camera_name.get().replace(' ', '_')
            messagebox.showinfo("Success", f"Calibration successful!\nUsed {len(self.captured_images)} images.\nRMS Error: {ret:.4f}\n\nAuto-exported to calibration/{cam_name_safe}/")
        except Exception as e:
            self.lbl_status.config(text=f"Status: Calibration failed.", foreground="#dc3545")
            messagebox.showerror("Calibration Error", f"Failed to calibrate camera:\n{str(e)}")

    def update_alpha(self, event=None):
        if self.calibration_results and self.img_size:
            mtx = np.array(self.calibration_results['camera_matrix'])
            dist = np.array(self.calibration_results['distortion_coefficients'])
            alpha = self.alpha_var.get()
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, self.img_size, alpha, self.img_size)
            self.new_camera_mtx = newcameramtx
            self.roi = roi
            self.calibration_results['new_camera_matrix'] = newcameramtx.tolist()
            self.calibration_results['roi'] = roi
            # The video loop will automatically use the new matrix on the next frame
            
    def update_frustum(self, event=None):
        if not self.calibration_results:
            return
        try:
            Z = self.calc_z.get()
            mtx = np.array(self.calibration_results['camera_matrix'])
            fx = mtx[0, 0]
            fy = mtx[1, 1]
            w, h = self.calibration_results['image_size']
            
            W_vis = Z * (w / fx)
            H_vis = Z * (h / fy)
            Area = W_vis * H_vis
            
            self.lbl_calc_w.config(text=f"Width: {W_vis:.2f} mm")
            self.lbl_calc_h.config(text=f"Height: {H_vis:.2f} mm")
            self.lbl_calc_area.config(text=f"Area: {Area:.2f} mm²")
        except ValueError:
            pass # invalid input

    def export_json(self):
        if not self.calibration_results: return
        cam_name_safe = self.camera_name.get().replace(' ', '_')
        timestamp = self.calibration_results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        dir_path = os.path.join("calibration", cam_name_safe)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{cam_name_safe}_{timestamp}_calibration.json")
        
        with open(path, 'w') as f:
            json.dump(self.calibration_results, f, indent=4)
        self.lbl_status.config(text=f"Status: Exported JSON to {path}")

    def export_report(self):
        if not self.calibration_results: return
        cam_name_safe = self.camera_name.get().replace(' ', '_')
        timestamp = self.calibration_results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        dir_path = os.path.join("calibration", cam_name_safe)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{cam_name_safe}_{timestamp}_report.md")
        
        mtx = np.array(self.calibration_results['camera_matrix'])
        dist = np.array(self.calibration_results['distortion_coefficients'])
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]
        w, h = self.calibration_results['image_size']
        
        # Calculate frustum area at various distances (1mm to 100mm)
        distances = list(range(1, 101))
        rows_count = 25
        cols_count = 4
        
        frustum_rows = []
        
        # Build headers with separator columns
        header_parts = [" Z(mm) | W(mm) | H(mm) | Area(mm²) "] * cols_count
        header = "|" + "|   |".join(header_parts) + "|"
        
        # Build markdown separators
        sep_parts = ["---|---|---|---"] * cols_count
        separator = "|" + "|---|".join(sep_parts) + "|"
        
        frustum_rows.append(header)
        frustum_rows.append(separator)
        
        for r in range(rows_count):
            row_sets = []
            for c in range(cols_count):
                idx = c * rows_count + r
                if idx < len(distances):
                    Z = distances[idx]
                    W_vis = Z * (w / fx)
                    H_vis = Z * (h / fy)
                    Area = W_vis * H_vis
                    row_sets.append(f" {Z} | {W_vis:.1f} | {H_vis:.1f} | {Area:.1f} ")
                else:
                    row_sets.append(" | | | ")
            frustum_rows.append("|" + "|   |".join(row_sets) + "|")
            
        frustum_table = "\n".join(frustum_rows)
        
        dist_flat = np.zeros(5)
        dist_raw = np.array(self.calibration_results['distortion_coefficients']).flatten()
        dist_flat[:min(5, len(dist_raw))] = dist_raw[:min(5, len(dist_raw))]
        
        md_content = f"""# Camera Calibration Report
**Camera Name:** {self.calibration_results['camera_name']}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## General Settings
| Parameter | Value |
| --- | --- |
| **FOV Configured** | {self.calibration_results['fov']} degrees |
| **Image Size (W x H)** | {w} x {h} pixels |
| **Valid Images Used** | {self.calibration_results['valid_images_used']} / {self.calibration_results['total_images']} |
| **RMS Reprojection Error** | {self.calibration_results['rms_error']:.4f} pixels |

## Intrinsics

The intrinsic parameters represent the internal optics of the camera.

### Camera Matrix ($K$)
The camera matrix defines the focal lengths ($f_x, f_y$) and the principal point ($c_x, c_y$):
$$
K = \\begin{{bmatrix}}
{fx:.2f} & 0 & {cx:.2f} \\\\
0 & {fy:.2f} & {cy:.2f} \\\\
0 & 0 & 1
\\end{{bmatrix}}
$$

### Distortion Coefficients ($D$)
The distortion coefficients $(k_1, k_2, p_1, p_2, k_3)$ model the radial and tangential lens distortion:
$$
D = \\begin{{bmatrix}} {dist_flat[0]:.5f} & {dist_flat[1]:.5f} & {dist_flat[2]:.5f} & {dist_flat[3]:.5f} & {dist_flat[4]:.5f} \\end{{bmatrix}}
$$

## Frustum (Visible Area) Calculator

The "Frustum" refers to the 3D region of space that is visible to the camera. Because the camera projects a 3D world onto a 2D sensor, the visible area expands as you move further away, forming a rectangular pyramid.

We calculate the physical size of this visible area at any given distance ($Z$) using the principle of similar triangles, based on the camera's intrinsic focal lengths ($f_x, f_y$) and its sensor resolution ($W_{{image}}, H_{{image}}$).

### Variables
*   **$Z$ (Distance)**: The perpendicular physical distance from the camera lens to the object plane (in millimeters).
*   **$f_x, f_y$ (Focal Length)**: The focal lengths of the camera in pixel units, calculated accurately during the calibration process.
*   **$W_{{image}}, H_{{image}}$ (Resolution)**: The native pixel resolution of the camera sensor.
*   **$W_{{visible}}, H_{{visible}}$ (Physical Area)**: The calculated physical width and height of the camera's field of view at distance $Z$ (in millimeters).

### Equations
$$ W_{{visible}} = Z \\cdot \\frac{{W_{{image}}}}{{f_x}} $$
$$ H_{{visible}} = Z \\cdot \\frac{{H_{{image}}}}{{f_y}} $$
$$ Area = W_{{visible}} \\cdot H_{{visible}} $$

### Visibility Table (1mm to 100mm)
{frustum_table}
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        self.lbl_status.config(text=f"Status: Exported Markdown Report to {path}")

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraCalibrationApp(root)
    root.mainloop()
