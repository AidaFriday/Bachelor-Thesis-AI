from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox,
    QFileDialog, QMessageBox
)
import os
import json

SETTINGS_FILE = "settings.json"


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_path = None
        self.model_name = "arcface"

        # Try to load saved settings
        self.load_settings()

        layout = QVBoxLayout()

        # --- Dataset selection ---
        self.dataset_label = QLabel("Choose dataset folder:")
        if self.dataset_path:
            self.dataset_label.setText(f"Dataset: {self.dataset_path}")
        self.dataset_btn = QPushButton("Browse Dataset")
        self.dataset_btn.clicked.connect(self.browse_dataset)

        # --- Model selection ---
        self.model_label = QLabel("Select model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["arcface", "facenet", "magface"])
        # set current selection from saved settings
        idx = self.model_combo.findText(self.model_name)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.currentTextChanged.connect(self.update_model)

        # --- Save button ---
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)

        # Layout
        layout.addWidget(self.dataset_label)
        layout.addWidget(self.dataset_btn)
        layout.addSpacing(15)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addStretch()
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    def browse_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", os.getcwd()
        )
        if path:
            self.dataset_path = path
            self.dataset_label.setText(f"Dataset: {path}")

    def update_model(self, text):
        """Update model_name whenever combo changes."""
        self.model_name = text

    def save_settings(self):
        """Save settings to JSON file and show confirmation."""
        data = {
            "model": self.model_name,
            "dataset": self.dataset_path,
        }
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)

        msg = (
            f"✅ Settings saved:\n\n"
            f"Model: {self.model_name}\nDataset: {self.dataset_path or 'Not selected'}"
        )
        QMessageBox.information(self, "Settings", msg)

    def load_settings(self):
        """Load settings from JSON if exists."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                self.model_name = data.get("model", "arcface")
                self.dataset_path = data.get("dataset", None)
                print(f"[INFO] Loaded settings: model={self.model_name}, dataset={self.dataset_path}")
            except Exception as e:
                print(f"[WARN] Could not load settings.json: {e}")
