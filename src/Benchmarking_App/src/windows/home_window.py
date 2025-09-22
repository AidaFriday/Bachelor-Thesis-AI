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
    QWidget,
    QComboBox,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from connector import load_model


class HomeWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition Benchmark App")
        self.setGeometry(100, 100, 900, 700)

        # Dropdown for model selection
        self.model_selector = QComboBox()
        self.model_selector.addItems(["arcface", "facenet", "magface"])

        # Buttons
        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setEnabled(False)

        # Video display
        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.model_selector)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.video_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Camera + wrapper
        self.cap = None
        self.wrapper = None

        # Timer for video updates
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
