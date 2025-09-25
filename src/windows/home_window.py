# ==== windows/home_window.py ====

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QComboBox,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from connector import load_model
from components.sidebar import SideBar


class HomeWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition Benchmark App")
        self.setGeometry(100, 100, 900, 700)

         # === Right-side content widgets ===
        self.model_selector = QComboBox()
        self.model_selector.addItems(["arcface", "facenet", "magface"])

        # Buttons
        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setEnabled(False)

        # Video display
        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)

        # === Layout for right side ===
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.model_selector)
        right_layout.addWidget(self.start_btn)
        right_layout.addWidget(self.stop_btn)
        right_layout.addWidget(self.video_label)

        right_container = QWidget()
        right_container.setLayout(right_layout)

        # Sidebar
        self.sidebar = SideBar()
        self.toggle_btn = QPushButton ("☰ Menu") # hamburger button
        self.toggle_btn.setFixedHeight(40)
        self.toggle_btn.clicked.connect(self.sidebar.toggle)

         # Place toggle + content vertically
        wrapper_layout = QVBoxLayout()
        wrapper_layout.addWidget(self.toggle_btn, alignment=Qt.AlignLeft)
        wrapper_layout.addWidget(right_container)

        wrapper = QWidget()
        wrapper.setLayout(wrapper_layout)

        # === Main layout (sidebar + right content) ===
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(wrapper)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Camera handling
        self.cap = None
        self.wrapper = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Connect buttons
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)

    def start_camera(self):
        model_name = self.model_selector.currentText()
        self.wrapper = load_model(model_name)
        print(f"[INFO] Loaded model: {self.wrapper.name}")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.setText("[ERROR] Cannot open camera")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(30)  # update ~30fps

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.setText("Camera stopped.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return

        # Detect + embed
        faces = self.wrapper.detect_and_embed(frame)
        disp = frame.copy()
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for px, py in f["kps"].astype(int):
                cv2.circle(disp, (px, py), 2, (0, 255, 255), -1)

        # Convert to Qt image
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
