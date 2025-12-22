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

# ✅ KEEP THIS (correct)
from components.embeddings_creation.dataset_cache import DatasetEmbeddingCache

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
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)

        # Apply theme
        self.apply_theme(self.settings_page.theme)

    # ------------------------------------------------------------------

    def start_camera(self):
        model_name = self.settings_page.model_name
        self.wrapper = load_model(model_name)
        print(f"[INFO] Loaded model: {self.wrapper.name}")

        # -------------------------------------------------------
        # Automatic dataset embeddings (model-adaptive)
        # -------------------------------------------------------
        try:
            self.face_db = DatasetEmbeddingCache(
                self.wrapper,
                self.settings_page.dataset_path
            )
            self.face_db.load_or_build()
            print(f"[INFO] Dataset ready for model: {self.wrapper.name}")
            print(f"[INFO] People in DB: {self.face_db.names}")
        except Exception as e:
            print(f"[ERROR] Failed to prepare dataset embeddings: {e}")
            self.face_db = None

        print("[INFO] Searching for cameras...")

        # -------------------------------------------------------
        # Detect OBS placeholder (blue screen with logo)
        # -------------------------------------------------------
        def looks_like_obs_placeholder(frame):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Blue-ish range of OBS placeholder background
            lower_blue = np.array([90, 40, 40])
            upper_blue = np.array([130, 255, 255])

            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            blue_ratio = mask.mean()

            # OBS placeholder = majority blue pixels
            return blue_ratio > 50

        # -------------------------------------------------------
        # 1) Try external camera (index 1)
        # -------------------------------------------------------
        external_cam_valid = False
        cap1 = cv2.VideoCapture(1)

        if cap1.isOpened():
            ret, frame = cap1.read()
            if ret:
                if not looks_like_obs_placeholder(frame):
                    external_cam_valid = True
                    selected_cam = 1
                    print("[INFO] External camera appears REAL → using index 1")
                else:
                    print("[INFO] External camera shows OBS placeholder → skipping.")
            else:
                print("[INFO] External camera opened but no frame → skipping.")
        else:
            print("[INFO] External camera (index 1) not opened.")

        cap1.release()

        # -------------------------------------------------------
        # 2) Fallback to laptop camera (index 0)
        # -------------------------------------------------------
        if not external_cam_valid:
            selected_cam = 0
            print("[INFO] Using laptop camera (index 0)")

        # -------------------------------------------------------
        # 3) Open selected camera
        # -------------------------------------------------------
        self.cap = cv2.VideoCapture(selected_cam)
        if not self.cap.isOpened():
            self.video_label.setText("[ERROR] Cannot open ANY camera!")
            print("[ERROR] Camera failed to open!")
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

        ok, frame = self.cap.read()
        if not ok:
            return

        disp = frame.copy()

        # ✅ ONE detection step (shared SCRFD detector)
        detections = self.wrapper.detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            kps = det["kps"]

            # ---- Align face for CURRENT model ----
            aligned = self.wrapper.detector.align_for(frame, kps)
            if aligned is None:
                continue

            # ---- Embed ----
            emb = self.wrapper.embed(aligned)
            if emb is None:
                continue

            # ---- Normalize (important for cosine) ----
            emb = emb.astype(np.float32)
            emb /= np.linalg.norm(emb)

            # ---- Recognition ----
            label_text = ""
            if self.face_db is not None:
                name, sim = self.face_db.match(emb)
                label_text = f"{name} | cos={sim:.3f}"

                # OPTIONAL threshold
                if sim < 0.45:
                    label_text = "Unknown"

            # ---- Draw bounding box ----
            color = self.model_colors.get(self.wrapper.name, (0, 255, 0))
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            # ---- Draw landmarks ----
            for px, py in kps.astype(int):
                cv2.circle(disp, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)

            # ---- Draw label ----
            if label_text:
                pos = (x1, max(20, y1 - 10))

                cv2.putText(
                    disp,
                    label_text,
                    pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    disp,
                    label_text,
                    pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # ---- Send frame to Qt ----
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------

    def toggle_sidebar(self):
        self.sidebar.toggle()
        self.toggle_btn.setText("☰" if self.sidebar._collapsed else "←")

    def apply_theme(self, theme: str):
        self.setStyleSheet(DARK_THEME if theme == "dark" else LIGHT_THEME)
        self.sidebar.apply_theme(theme)
