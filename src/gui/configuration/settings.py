import sys
from PyQt5.QtGui import QPalette

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import pyqtSignal, Qt
import os
import json

# Import the centralized dataset manager
from dataset.manager import DatasetManager

# Project root (…/src)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")


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
    theme_changed = pyqtSignal(str)  # 🔔 notify theme changes

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Core state ---
        self.model_name = "arcface"
        self.theme = "light"

        # Central dataset manager (handles registry & validation)
        self.dataset_manager = DatasetManager()
        self.dataset_name = None
        self.dataset_path = None

        # Load saved settings if available
        self.load_settings()

        # --- Layout ---
        layout = QVBoxLayout()

        # --- Dataset section (auto detection, no combo box) ---
        self.dataset_label = QLabel("Dataset:")
        self.dataset_name_label = QLabel(self.dataset_name or "Not selected")
        self.dataset_path_label = QLabel(f"Path: {self.dataset_path or 'Not selected'}")
        self.dataset_browse_btn = QPushButton("Browse Dataset Folder")
        self.dataset_browse_btn.clicked.connect(self.browse_dataset)

        # --- Model selection ---
        self.model_label = QLabel("Select model:")
        self.model_combo = QComboBox()

        # ✅ Load model list dynamically from model.config if available
        model_config_path = os.path.join(BASE_DIR, "models", "model.config")
        try:
            with open(model_config_path, "r") as f:
                model_config = json.load(f)
                model_names = list(model_config.keys())
        except Exception as e:
            print(f"[WARN] Could not read model.config: {e}")
            model_names = ["arcface", "facenet", "insightface"]

        self.model_combo.addItems(model_names)
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
        self.theme_toggle_btn = QPushButton("Toggle Light/Dark")
        self.theme_toggle_btn.clicked.connect(self.toggle_theme)

        # --- Save button ---
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)

        # --- Assemble layout ---
        layout.addWidget(self.dataset_label)
        layout.addWidget(self.dataset_name_label)
        layout.addWidget(self.dataset_path_label)
        layout.addWidget(self.dataset_browse_btn)
        layout.addSpacing(15)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addSpacing(15)
        layout.addWidget(self.theme_label)
        layout.addWidget(self.theme_combo)
        layout.addWidget(self.theme_toggle_btn)
        layout.addStretch()
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

    # ===============================================================
    # 🔹 Dataset Handling
    # ===============================================================
    def browse_dataset(self):
        """Manually browse for a dataset folder and auto-detect type."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", os.getcwd()
        )
        if not path:
            return

        self.dataset_path = path

        # Auto-detect dataset type from folder name or path
        lower = path.lower()
        if "ytf" in lower or "aligned_images_db" in lower:
            self.dataset_name = "ytf"
        elif "lfw" in lower:
            self.dataset_name = "lfw"
        else:
            self.dataset_name = "unknown"

        # Update UI
        self.dataset_name_label.setText(self.dataset_name.upper())
        self.dataset_path_label.setText(f"Path: {path}")

    # ===============================================================
    # 🔹 Model & Theme
    # ===============================================================
    def update_model(self, text):
        self.model_name = text

    def update_theme(self, text):
        self.theme = text
        self.theme_changed.emit(self.effective_theme())

    # ===============================================================
    # 🔹 Save & Load
    # ===============================================================
    # ===============================================================
    # 🔹 Save & Load
    # ===============================================================
    def save_settings(self):
        data = {
            "model": self.model_name,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "theme": self.theme,
        }

        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)

        # Normal confirmation message
        msg = (
            f"✅ Settings saved:\n\n"
            f"Model: {self.model_name}\n"
            f"Dataset: {self.dataset_name or 'Not selected'}\n"
            f"Path: {self.dataset_path or 'N/A'}\n"
            f"Theme: {self.theme}"
        )

        QMessageBox.information(self, "Settings", msg)

    def load_settings(self):
        """Load settings.json and prefill UI state."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)

                self.model_name = data.get("model", "arcface")
                self.dataset_name = data.get("dataset_name")
                self.dataset_path = data.get("dataset_path")
                self.theme = data.get("theme", "light")

                print(
                    f"[INFO] Loaded settings: "
                    f"model={self.model_name}, dataset={self.dataset_name}, "
                    f"path={self.dataset_path}, theme={self.theme}"
                )

                if self.dataset_name:
                    try:
                        self.dataset_manager.set_dataset(
                            self.dataset_name, self.dataset_path
                        )
                    except Exception as e:
                        print(f"[WARN] Could not set dataset: {e}")

            except Exception as e:
                print(f"[WARN] Could not load settings.json: {e}")

        # If widgets already exist, sync UI with loaded values
        if hasattr(self, "dataset_name_label"):
            self.dataset_name_label.setText(self.dataset_name or "Not selected")
        if hasattr(self, "dataset_path_label"):
            self.dataset_path_label.setText(
                f"Path: {self.dataset_path or 'Not selected'}"
            )
        if hasattr(self, "model_combo"):
            idx = self.model_combo.findText(self.model_name)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if hasattr(self, "theme_combo"):
            idx = self.theme_combo.findText(self.theme)
            if idx >= 0:
                self.theme_combo.setCurrentIndex(idx)

    def system_theme(self) -> str:
        """
        Best-effort system theme detection.
        Returns "dark" or "light".
        """
        # Works reasonably cross-platform by checking palette brightness
        pal = self.palette()
        base = pal.color(QPalette.Window).lightness()
        text = pal.color(QPalette.WindowText).lightness()

        # If window is dark and text is light -> dark theme
        return "dark" if base < text else "light"

    def effective_theme(self) -> str:
        """
        If user selected 'system', map to actual 'light'/'dark'.
        """
        return self.system_theme() if self.theme == "system" else self.theme
    
    def toggle_theme(self):
        # If currently 'system', toggle the *effective* theme and switch to explicit mode
        current = self.effective_theme()
        new_theme = "dark" if current == "light" else "light"

        self.theme = new_theme
        # Update combo selection
        idx = self.theme_combo.findText(new_theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)

        self.theme_changed.emit(new_theme)
