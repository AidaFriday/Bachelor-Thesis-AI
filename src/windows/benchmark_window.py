# ==== windows/benchmark_window.py ====
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton

class BenchmarkPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # --- Benchmark parameter buttons ---
        self.btn_accuracy   = QPushButton("Accuracy")
        self.btn_efficiency = QPushButton("Efficiency / Runtime Metrics")
        self.btn_resources  = QPushButton("Resource Usage")
        self.btn_memory     = QPushButton("Memory Consumption")   # placeholder
        self.btn_latency    = QPushButton("Latency")              # placeholder
        self.btn_energy     = QPushButton("Energy Usage")         # placeholder

        # Add them to layout
        for btn in [
            self.btn_accuracy,
            self.btn_efficiency,
            self.btn_resources,
            self.btn_memory,
            self.btn_latency,
            self.btn_energy,
        ]:
            btn.setMinimumHeight(40)
            layout.addWidget(btn)

        layout.addStretch()
        self.setLayout(layout)
