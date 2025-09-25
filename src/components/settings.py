# ==== components/settings.py ====
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QMessageBox
import os

class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_path = None
        self.model_name = "arcface"

        layout = QVBoxLayout()

        self.dataset_label = QLabel("Choose dataset folder:")
        self.dataset_btn = QPushButton("Browse Dataset")
        self.dataset_btn.clicked.connect(self.browse_dataset)

        self.model_label = QLabel("Select model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["arcface", "facenet", "magface"])

        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)

        layout.addWidget(self.dataset_label)
        layout.addWidget(self.dataset_btn)
        layout.addSpacing(15)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addStretch()
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    def browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", os.getcwd())
        if path:
            self.dataset_path = path
            self.dataset_label.setText(f"Dataset: {path}")

    def save_settings(self):
        self.model_name = self.model_combo.currentText()
        msg = f"✅ Settings saved:\n\nModel: {self.model_name}\nDataset: {self.dataset_path or 'Not selected'}"
        QMessageBox.information(self, "Settings", msg)
