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
    background: #ffffff;
    color: #111827;
    font-size: 13px;
}

/* Labels */
QLabel {
    color: #111827;
    font-weight: 500;
}

/* Inputs */
QComboBox {
    background: rgba(0,0,0,0.03);
    border: 1px solid rgba(0,0,0,0.10);
    border-radius: 12px;
    padding: 8px 12px;
    min-height: 34px;
}
QComboBox:hover {
    border: 1px solid rgba(0,0,0,0.18);
    background: rgba(0,0,0,0.045);
}
QComboBox:focus {
    border: 1px solid rgba(10,163,127,0.65);
}

/* Primary button */
QPushButton#PrimaryButton {
    background: rgba(10,163,127,0.14);
    border: 1px solid rgba(10,163,127,0.25);
    border-radius: 14px;
    padding: 10px 14px;
    font-weight: 600;
}
QPushButton#PrimaryButton:hover {
    background: rgba(10,163,127,0.18);
    border: 1px solid rgba(10,163,127,0.35);
}
QPushButton#PrimaryButton:pressed {
    background: rgba(10,163,127,0.24);
}

/* Secondary button */
QPushButton#SecondaryButton {
    background: rgba(0,0,0,0.03);
    border: 1px solid rgba(0,0,0,0.10);
    border-radius: 14px;
    padding: 10px 14px;
    font-weight: 600;
}
QPushButton#SecondaryButton:hover {
    background: rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.18);
}
QPushButton#SecondaryButton:pressed {
    background: rgba(0,0,0,0.08);
}

/* Disabled */
QPushButton:disabled {
    color: rgba(17,24,39,0.35);
    background: rgba(0,0,0,0.03);
    border: 1px solid rgba(0,0,0,0.06);
}
"""

DARK_THEME = """
QWidget {
    background: #1f1f23;
    color: #e5e7eb;
    font-size: 13px;
}

/* Labels */
QLabel {
    color: #e5e7eb;
    font-weight: 500;
}

/* Inputs */
QComboBox {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 8px 12px;
    min-height: 34px;
    color: #e5e7eb;
}
QComboBox:hover {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
}
QComboBox:focus {
    border: 1px solid rgba(158,240,200,0.55);
}

/* Dropdown list */
QAbstractItemView {
    background: #1f1f23;
    color: #e5e7eb;
    border: 1px solid rgba(255,255,255,0.10);
    selection-background-color: rgba(158,240,200,0.18);
}

/* Primary button */
QPushButton#PrimaryButton {
    background: rgba(158,240,200,0.14);
    border: 1px solid rgba(158,240,200,0.22);
    border-radius: 14px;
    padding: 10px 14px;
    font-weight: 600;
    color: #e5e7eb;
}
QPushButton#PrimaryButton:hover {
    background: rgba(158,240,200,0.18);
    border: 1px solid rgba(158,240,200,0.32);
}
QPushButton#PrimaryButton:pressed {
    background: rgba(158,240,200,0.24);
}

/* Secondary button */
QPushButton#SecondaryButton {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 10px 14px;
    font-weight: 600;
    color: #e5e7eb;
}
QPushButton#SecondaryButton:hover {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
}
QPushButton#SecondaryButton:pressed {
    background: rgba(255,255,255,0.12);
}

/* Disabled */
QPushButton:disabled {
    color: rgba(229,231,235,0.35);
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
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
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

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
        self.theme_toggle_btn.setObjectName("SecondaryButton")
        self.theme_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.theme_toggle_btn.clicked.connect(self.toggle_theme)

        # --- Save button ---
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setObjectName("PrimaryButton")
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.clicked.connect(self.save_settings)

        # --- Assemble layout ---
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addSpacing(12)
        layout.addWidget(self.theme_label)
        layout.addWidget(self.theme_combo)
        layout.addWidget(self.theme_toggle_btn)
        layout.addStretch()
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

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
    def save_settings(self):
        data = {
            "model": self.model_name,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "theme": self.theme,
        }

        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)

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
        if hasattr(self, "model_combo"):
            idx = self.model_combo.findText(self.model_name)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if hasattr(self, "theme_combo"):
            idx = self.theme_combo.findText(self.theme)
            if idx >= 0:
                self.theme_combo.setCurrentIndex(idx)

    def system_theme(self) -> str:
        pal = self.palette()
        base = pal.color(QPalette.Window).lightness()
        text = pal.color(QPalette.WindowText).lightness()
        return "dark" if base < text else "light"

    def effective_theme(self) -> str:
        return self.system_theme() if self.theme == "system" else self.theme

    def toggle_theme(self):
        current = self.effective_theme()
        new_theme = "dark" if current == "light" else "light"

        self.theme = new_theme
        idx = self.theme_combo.findText(new_theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)

        self.theme_changed.emit(new_theme)
