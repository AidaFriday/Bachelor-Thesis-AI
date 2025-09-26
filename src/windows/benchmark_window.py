from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget, QFrame
from PyQt5.QtCore import Qt


class BenchmarkPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Main vertical layout ---
        main_layout = QVBoxLayout()

        # --- Row of buttons (horizontal) ---
        button_layout = QHBoxLayout()
        self.btn_accuracy   = QPushButton("Accuracy")
        self.btn_efficiency = QPushButton("Efficiency / Runtime Metrics")
        self.btn_resources  = QPushButton("Resource Usage")
        self.btn_memory     = QPushButton("Memory Consumption")
        self.btn_latency    = QPushButton("Latency")
        self.btn_energy     = QPushButton("Energy Usage")

        for btn in [
            self.btn_accuracy,
            self.btn_efficiency,
            self.btn_resources,
            self.btn_memory,
            self.btn_latency,
            self.btn_energy,
        ]:
            btn.setMinimumHeight(40)
            button_layout.addWidget(btn)

        main_layout.addLayout(button_layout)

        # --- Content area (stacked pages for each parameter) ---
        self.output_stack = QStackedWidget()
        self.pages = {}

        def add_page(name, text):
            page = QWidget()
            layout = QVBoxLayout()
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            page.setLayout(layout)
            idx = self.output_stack.addWidget(page)
            self.pages[name] = idx

        add_page("accuracy",   "Accuracy results will be shown here.")
        add_page("efficiency", "Efficiency / runtime metrics here.")
        add_page("resources",  "Resource usage analysis here.")
        add_page("memory",     "Memory consumption results here.")
        add_page("latency",    "Latency measurements here.")
        add_page("energy",     "Energy usage statistics here.")

        # Wrap in QFrame for border
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                border: 2px solid #800080;   /* purple border */
                border-radius: 6px;
                background-color: #fafafa;
            }
        """)
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.output_stack)
        frame.setLayout(frame_layout)

        main_layout.addWidget(frame)
        self.setLayout(main_layout)

        
        # --- Connect buttons to pages ---
        self.btn_accuracy.clicked.connect(lambda: self.show_page("accuracy"))
        self.btn_efficiency.clicked.connect(lambda: self.show_page("efficiency"))
        self.btn_resources.clicked.connect(lambda: self.show_page("resources"))
        self.btn_memory.clicked.connect(lambda: self.show_page("memory"))
        self.btn_latency.clicked.connect(lambda: self.show_page("latency"))
        self.btn_energy.clicked.connect(lambda: self.show_page("energy"))

        # Start with no selection
        self.output_stack.setCurrentIndex(-1)

    def show_page(self, name):
        idx = self.pages.get(name, -1)
        if idx >= 0:
            self.output_stack.setCurrentIndex(idx)
