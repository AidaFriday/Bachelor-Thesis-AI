from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import pyqtSignal
import os
import json

# Project root (…/src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SCRIPTS_DIR = os.path.join(BASE_DIR, "dataset")


SETTINGS_FILE = "settings.json"

LIGHT_THEME = """
    QWidget {
        background-color: #ffffff;
        color: #000000;
    }
    QPushButton {
        background-color: #f0f0f0;
        border: 1px solid #888;
        padding: 6px;
    }
    QPushButton:hover {
        background-color: #e0e0e0;
    }
    QLabel {
        color: #000000;
    }
"""

DARK_THEME = """
    QWidget {
        background-color: #2d2d2d;
        color: #dddddd;
    }
    QPushButton {
        background-color: #444;
        border: 1px solid #888;
        padding: 6px;
        color: #ffffff;
    }
    QPushButton:hover {
        background-color: #555;
    }
    QLabel {
        color: #dddddd;
    }
"""


class SettingsPage(QWidget):
    theme_changed = pyqtSignal(str)  # 🔔 signal to notify theme changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_path = None
        self.model_name = "arcface"
        self.theme = "light"

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
        self.model_combo.addItems(["arcface", "facenet", "insightface"])
        idx = self.model_combo.findText(self.model_name)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.currentTextChanged.connect(self.update_model)

        # --- Theme selection ---
        self.theme_label = QLabel("Select theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["light", "dark"])
        idx = self.theme_combo.findText(self.theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        self.theme_combo.currentTextChanged.connect(self.update_theme)

        # --- Save button ---
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)

        # Layout
        layout.addWidget(self.dataset_label)
        layout.addWidget(self.dataset_btn)
        layout.addSpacing(15)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addSpacing(15)
        layout.addWidget(self.theme_label)
        layout.addWidget(self.theme_combo)
        layout.addStretch()
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    def _resolve_dataset_script(self, selected_path: str):
        """Given a selected dataset folder, return (folder_name, matching_script_path_or_None).
        Matching is case-insensitive and ignores spaces and hyphens.
        """
        ds_name = os.path.basename(os.path.normpath(selected_path))
        normalized = ds_name.replace(" ", "").replace("-", "").lower()

        try:
            candidates = {
                os.path.splitext(f)[0]
                .replace(" ", "")
                .replace("-", "")
                .lower(): os.path.join(DATASET_SCRIPTS_DIR, f)
                for f in os.listdir(DATASET_SCRIPTS_DIR)
                if f.endswith(".py")
            }
        except Exception:
            candidates = {}

        return ds_name, candidates.get(normalized)

    def browse_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", os.getcwd()
        )
        if not path:
            return

        ds_name, script_path = self._resolve_dataset_script(path)

        if not script_path or not os.path.exists(script_path):
            QMessageBox.critical(
                self,
                "Dataset not supported",
                (
                    f"Dataset logic not implemented for '{ds_name}'.\n\n"
                    f"I expected a Python file in:\n"
                    f"{DATASET_SCRIPTS_DIR}\n"
                    f"named like '{ds_name}.py' (case-insensitive).\n\n"
                    "Create the dataset loader there and try again."
                ),
            )
            return

        # ok ✔️
        self.dataset_path = path
        self.dataset_label.setText(f"Dataset: {path}")

    def update_model(self, text):
        self.model_name = text

    def update_theme(self, text):
        self.theme = text
        self.theme_changed.emit(text)  # 🔔 tell parent window

    def save_settings(self):
        data = {
            "model": self.model_name,
            "dataset": self.dataset_path,
            "theme": self.theme,
        }
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)

        msg = (
            f"✅ Settings saved:\n\n"
            f"Model: {self.model_name}\n"
            f"Dataset: {self.dataset_path or 'Not selected'}\n"
            f"Theme: {self.theme}"
        )
        QMessageBox.information(self, "Settings", msg)

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                self.model_name = data.get("model", "arcface")
                self.dataset_path = data.get("dataset", None)
                self.theme = data.get("theme", "light")
                print(
                    f"[INFO] Loaded settings: model={self.model_name}, dataset={self.dataset_path}, theme={self.theme}"
                )
            except Exception as e:
                print(f"[WARN] Could not load settings.json: {e}")
