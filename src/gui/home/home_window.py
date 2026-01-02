# home_window.py

from pathlib import Path
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QStackedWidget,
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap

import cv2
from connector import load_model
from components.sidebar import SideBar
from gui.configuration.settings import SettingsPage, LIGHT_THEME, DARK_THEME
from gui.benchmark.benchmark_window import BenchmarkPage
from components.embeddings_creation.dataset_cache import DatasetEmbeddingCache
import time
from components.utilities.live_fps_latency import (
    LiveFpsLatency,
    BenchmarkMetricsProvider,
    format_metric,
)
from components.utilities.live_memory_usage import LiveMemoryUsage


class HomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition Benchmark App")
        self.setGeometry(100, 100, 900, 700)

        # --- Stacked pages ---
        self.stacked = QStackedWidget()

        # Page 1 – Home (Camera UI)
        self.home_page = QWidget()
        home_layout = QVBoxLayout()

        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setEnabled(False)

        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(800, 500)
        self.video_label.setScaledContents(True)

        home_layout.addWidget(self.start_btn)
        home_layout.addWidget(self.stop_btn)
        home_layout.addWidget(self.video_label)
        self.home_page.setLayout(home_layout)

        # Page 2 – Settings
        self.settings_page = SettingsPage()

        # --- Model colors (consistent) ---
        self.model_colors = {
            "arcface": (0, 180, 255),  # cyan
            "facenet": (0, 255, 0),  # green
            "adaface": (180, 0, 255),  # magenta
            "facenet_camera": (255, 140, 0),  # orange
            "facenet_original": (0, 255, 0),
        }

        self.settings_page.theme_changed.connect(self.apply_theme)

        # Page 3 – Benchmark
        self.benchmark_page = BenchmarkPage(
            get_model_name=lambda: self.settings_page.model_name
        )

        # Add pages
        self.stacked.addWidget(self.home_page)
        self.stacked.addWidget(self.settings_page)
        self.stacked.addWidget(self.benchmark_page)

        # Sidebar + button
        self.sidebar = SideBar()
        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setFixedSize(QSize(40, 40))
        self.toggle_btn.clicked.connect(self.toggle_sidebar)

        # Navigation
        self.sidebar.btn_home.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.sidebar.btn_settings.clicked.connect(
            lambda: self.stacked.setCurrentIndex(1)
        )
        self.sidebar.btn_benchmark.clicked.connect(
            lambda: self.stacked.setCurrentIndex(2)
        )

        # Layout wrapper
        wrapper_layout = QVBoxLayout()
        wrapper_layout.addWidget(self.toggle_btn, alignment=Qt.AlignLeft)
        wrapper_layout.addWidget(self.stacked)

        wrapper = QWidget()
        wrapper.setLayout(wrapper_layout)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(wrapper)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Camera
        self.cap = None
        self.wrapper = None
        self.face_db = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)

        # Apply theme
        self.apply_theme(self.settings_page.theme)

    # ------------------------------------------------------------------
    def start_camera(self):
        model_name = self.settings_page.model_name

        # ---- Load model & DB (keep your existing code here) ----
        self.wrapper = load_model(model_name)

        # ---- Camera selection ----
        def looks_like_obs_placeholder(frame):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([90, 40, 40])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            return mask.mean() > 50

        selected_cam = 0
        cap1 = cv2.VideoCapture(1)
        if cap1.isOpened():
            ret, frame = cap1.read()
            if ret and not looks_like_obs_placeholder(frame):
                selected_cam = 1
                print("[INFO] External camera appears REAL → using index 1")
        cap1.release()

        # ---- Open camera ONCE ----
        self.cap = cv2.VideoCapture(selected_cam)
        if not self.cap.isOpened():
            self.video_label.setText("[ERROR] Cannot open camera")
            return

        print(f"[INFO] Camera started: index={selected_cam}")

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.timer.start(30)

    # ------------------------------------------------------------------

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[INFO] Camera stopped.")
        self.video_label.setText("Camera stopped.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------

    def update_frame(self):
        if self.cap is None:
            return

        t0 = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok:
            return

        disp = frame.copy()
        detections = self.wrapper.detector.detect(frame)

        def _extract_bbox_kps(det):
            if isinstance(det, dict):
                bbox = det.get("bbox")
                kps = det.get("kps")
            else:
                bbox = getattr(det, "bbox", None)
                kps = getattr(det, "kps", None)
            if bbox is None or kps is None:
                return None, None
            return np.array(bbox).astype(int), np.array(kps)

        for det in detections:
            bbox, kps = _extract_bbox_kps(det)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            aligned = self.wrapper.detector.align_for(frame, kps)
            if aligned is None:
                continue

            emb = self.wrapper.embed(aligned)
            if emb is None:
                continue

            emb = emb.astype(np.float32)
            n = np.linalg.norm(emb)
            if n > 0:
                emb /= n

            label_text = ""
            if self.face_db:
                name, sim = self.face_db.match(emb)
                label_text = f"{name} | cos={sim:.3f}" if sim >= 0.65 else "Unknown"

            color = self.model_colors.get(self.wrapper.name, (0, 255, 0))
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 1)

            for px, py in kps.astype(int):
                cv2.circle(disp, (px, py), 2, (0, 255, 255), -1)

            if label_text:
                cv2.putText(
                    disp,
                    label_text,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        # -------------------------------
        # PERF + MEMORY METRICS
        # -------------------------------
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self.perf.tick_latency(latency_ms)
        self.perf.tick_frame()

        self.mem.tick()

        fps = self.perf.mean_fps()
        lat = self.perf.mean_latency_ms()
        ram = self.mem.mean_rss_mb()
        model_ram = self.mem.model_rss_delta_mb()

        lines = [
            f"FPS: {format_metric(fps, 'fps')}",
            f"Latency: {format_metric(lat, 'ms')}",
            f"RAM: {ram:.1f} MB",
            f"Model RAM: {model_ram:.1f} MB",
        ]

        H, W = disp.shape[:2]
        x, y0, dy = W - 300, 30, 34

        for i, txt in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            y = y0 + i * dy
            cv2.rectangle(
                disp, (x - 10, y - th - 10), (x + tw + 10, y + 6), (0, 0, 0), -1
            )
            cv2.putText(
                disp, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.video_label.setPixmap(
            QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
        )

    # ------------------------------------------------------------------

    def toggle_sidebar(self):
        self.sidebar.toggle()
        self.toggle_btn.setText("☰" if self.sidebar._collapsed else "←")

    def apply_theme(self, theme: str):
        self.setStyleSheet(DARK_THEME if theme == "dark" else LIGHT_THEME)
        self.sidebar.apply_theme(theme)
