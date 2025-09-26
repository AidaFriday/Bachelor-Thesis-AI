# ==== windows/home_window.py ====
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QComboBox, QStackedWidget
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap

import cv2
from connector import load_model
from components.sidebar import SideBar
from components.settings import SettingsPage   
from windows.benchmark_window import BenchmarkPage


class HomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition Benchmark App")
        self.setGeometry(100, 100, 900, 700)

        # --- Stacked pages ---
        self.stacked = QStackedWidget()

        # Page 1: Home (camera UI)
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

        # Page 2: Settings
        self.settings_page = SettingsPage()
        # Page 3: Benchmark
        self.benchmark_page = BenchmarkPage()
        self.stacked.addWidget(self.benchmark_page)  # index 2

        # Add pages to stacked
        self.stacked.addWidget(self.home_page)     # index 0
        self.stacked.addWidget(self.settings_page) # index 1
        self.stacked.addWidget(self.benchmark_page)  # index 2

        # Sidebar
        self.sidebar = SideBar()
        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setFixedSize(QSize(40, 40))  # square button
        self.toggle_btn.clicked.connect(self.toggle_sidebar)

        # Sidebar navigation
        self.sidebar.btn_home.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.sidebar.btn_settings.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        self.sidebar.btn_benchmark.clicked.connect(lambda: self.stacked.setCurrentIndex(2)) 

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

    def start_camera(self):
        # get model from Settings page
        model_name = self.settings_page.model_name
        self.wrapper = load_model(model_name)
        print(f"[INFO] Loaded model: {self.wrapper.name}")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.setText("[ERROR] Cannot open camera")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(30)


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

        faces = self.wrapper.detect_and_embed(frame)
        disp = frame.copy()
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for px, py in f["kps"].astype(int):
                cv2.circle(disp, (px, py), 2, (0, 255, 255), -1)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def toggle_sidebar(self):
        """Toggle sidebar and update button text."""
        self.sidebar.toggle()
        if self.sidebar._collapsed:
            self.toggle_btn.setText("☰")   # burger
        else:
            self.toggle_btn.setText("←")   # back arrow
