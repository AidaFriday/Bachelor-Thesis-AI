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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

        # --- Dataset selection ---
        self.dataset_label = QLabel("Select dataset:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(self.dataset_manager.list_available())

        # Set current dataset if saved
        if self.dataset_name:
            idx = self.dataset_combo.findText(self.dataset_name, Qt.MatchFixedString)
            if idx >= 0:
                self.dataset_combo.setCurrentIndex(idx)

        # Keep dataset_name synced
        self.dataset_name = self.dataset_combo.currentText()
        self.dataset_combo.currentTextChanged.connect(self.update_dataset_choice)

        self.dataset_path_label = QLabel(f"Path: {self.dataset_path or 'Not selected'}")
        self.dataset_browse_btn = QPushButton("Browse Dataset Folder")
        self.dataset_browse_btn.clicked.connect(self.browse_dataset)

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

        # --- Assemble layout ---
        layout.addWidget(self.dataset_label)
        layout.addWidget(self.dataset_combo)
        layout.addWidget(self.dataset_path_label)
        layout.addWidget(self.dataset_browse_btn)
        layout.addSpacing(15)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addSpacing(15)
        layout.addWidget(self.theme_label)
        layout.addWidget(self.theme_combo)
        layout.addStretch()
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

    # ===============================================================
    # 🔹 Dataset Handling
    # ===============================================================
    def update_dataset_choice(self, name: str):
        """Update selected dataset name and apply default path."""
        self.dataset_name = name
        default_info = self.dataset_manager.DATASET_REGISTRY.get(name.lower(), {})
        default_path = default_info.get("default_path")
        self.dataset_path = default_path
        self.dataset_path_label.setText(f"Path: {self.dataset_path or 'N/A'}")

    def browse_dataset(self):
        """Manually browse for a dataset folder and override default."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder", os.getcwd()
        )
        if not path:
            return

        # Apply manually chosen path to the currently selected dataset
        self.dataset_path = path
        self.dataset_path_label.setText(f"Path: {path}")

    # ===============================================================
    # 🔹 Model & Theme
    # ===============================================================
    def update_model(self, text):
        self.model_name = text

    def update_theme(self, text):
        self.theme = text
        self.theme_changed.emit(text)

    # ===============================================================
    # 🔹 Save & Load
    # ===============================================================

    def save_settings(self):
        # Always read current dataset from combo box in case signal didn't fire
        self.dataset_name = self.dataset_combo.currentText().strip() or None

        # --- Auto-correct dataset type based on chosen path ---
        auto_dataset = self.dataset_name
        if self.dataset_path and "ytf" in self.dataset_path.lower():
            auto_dataset = "ytf"
        elif self.dataset_path and "lfw" in self.dataset_path.lower():
            auto_dataset = "lfw"

        data = {
            "model": self.model_name,
            "dataset_name": auto_dataset,
            "dataset_path": self.dataset_path,
            "theme": self.theme,
        }

        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)

        QMessageBox.information(
            self,
            "Settings",
            (
                f"✅ Settings saved:\n\n"
                f"Model: {self.model_name}\n"
                f"Dataset: {auto_dataset or 'Not selected'}\n"
                f"Path: {self.dataset_path or 'N/A'}\n"
                f"Theme: {self.theme}"
            ),
        )

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

                # Pre-set dataset in manager if valid
                if self.dataset_name:
                    try:
                        self.dataset_manager.set_dataset(
                            self.dataset_name, self.dataset_path
                        )
                    except Exception as e:
                        print(f"[WARN] Could not set dataset: {e}")

            except Exception as e:
                print(f"[WARN] Could not load settings.json: {e}")
