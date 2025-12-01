# ==== windows/benchmark_window.py ====

import os
import sys
import subprocess
import json
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QFrame,
    QProgressBar,
    QInputDialog,
    QMessageBox,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QLabel,
)
from PyQt5.QtCore import QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from .handlers.handler_progress import handle_progress
from .handlers.handler_confusion import handle_confusion
from .handlers.handler_latency import handle_latency
from .handlers.handler_fps import handle_fps
from .handlers.handler_roc_image import handle_roc_image
from .handlers.handler_accuracy_image import handle_accuracy_image
from .handlers.handler_accuracy_video import handle_accuracy_video


# ------------------ Dialog for YTF subject selection ------------------
class SelectSubjectsDialog(QDialog):
    """Popup dialog to select multiple subjects (folders) from dataset."""

    def __init__(self, dataset_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select People")
        self.setMinimumWidth(400)
        self.selected_subjects = []

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)

        # Load all subfolders (people)
        if os.path.isdir(dataset_path):
            people = sorted(
                [
                    d
                    for d in os.listdir(dataset_path)
                    if os.path.isdir(os.path.join(dataset_path, d))
                ]
            )
            for name in people:
                item = QListWidgetItem(name)
                item.setCheckState(0)
                self.list_widget.addItem(item)
        else:
            print(f"[WARN] Invalid dataset path: {dataset_path}")

        layout.addWidget(self.list_widget)

        btns = QHBoxLayout()
        select_all_btn = QPushButton("All")
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")

        select_all_btn.clicked.connect(self._select_all)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        btns.addWidget(select_all_btn)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def accept(self):
        self.selected_subjects = [
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState()
        ]
        super().accept()

    def _select_all(self):
        # special code path: tells caller to process entire dataset
        self.selected_subjects = ["__ALL__"]
        super().accept()


# ------------------ Simple Metric Selection Dialog ------------------
class SelectMetricDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Evaluation Method")
        self.setFixedWidth(360)

        layout = QVBoxLayout(self)

        label = QLabel(
            "Which evaluation metric would you like to compute?\n\n"
            "• ROC Curve (AUC, EER, TAR@FAR)\n"
            "• Confusion Matrix (TP/TN/FP/FN, Accuracy, Threshold)"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        button_layout = QHBoxLayout()

        btn_roc = QPushButton("ROC Curve")
        btn_cm = QPushButton("Confusion Matrix")

        btn_roc.clicked.connect(lambda: self._choose("roc"))
        btn_cm.clicked.connect(lambda: self._choose("cm"))

        button_layout.addWidget(btn_roc)
        button_layout.addWidget(btn_cm)

        layout.addLayout(button_layout)
        self.selection = None

    def _choose(self, selection):
        self.selection = selection
        self.accept()


# ------------------ Worker Thread ------------------
class RunnerThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(
        self,
        file_path,
        model_name,
        dataset_path=None,
        test_image=None,
        iters=None,
        extra_args=None,
    ):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.test_image = test_image
        self.iters = iters
        self.extra_args = extra_args or []  # NEW

    def run(self):
        try:
            cmd = [
                sys.executable,
                "-u",
                self.file_path,
                "--model",
                self.model_name,
            ]

            # only scripts that expect it get --iters
            if self.iters is not None:
                cmd.extend(["--iters", str(self.iters)])

            if self.dataset_path:
                cmd.extend(["--dataset", self.dataset_path])

            if self.extra_args:
                cmd.extend(self.extra_args)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for line in process.stdout:
                self.output_signal.emit(line.strip())
            process.stdout.close()
            process.wait()
        except Exception as e:
            self.output_signal.emit(json.dumps({"error": str(e)}))


# ------------------ Benchmark Page ------------------
class BenchmarkPage(QWidget):
    def __init__(self, parent=None, get_model_name=None):
        super().__init__(parent)
        self.get_model_name = get_model_name

        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.benchmark_dir = os.path.join(base_dir, "benchmark_parameters")

        self.test_image = os.path.join(
            self.benchmark_dir, "validation_accuracy", "test_image", "test_image1.jpg"
        )

        # Load dataset from settings.json
        self.settings_path = os.path.join(base_dir, "settings.json")
        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r") as f:
                cfg = json.load(f)
            # Unified dataset key support
            self.dataset_path = (
                cfg.get("dataset_path")
                or cfg.get("dataset")
                or r"C:\programming\Datasets\LFW\lfw-deepfunneled"
            )
        else:
            self.dataset_path = r"C:\programming\Datasets\LFW\lfw-deepfunneled"

        main_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()
        main_layout.addLayout(self.button_layout)

        # Output frame
        self.output_stack = QStackedWidget()
        self.pages = {}

        frame = QFrame()
        frame.setStyleSheet(
            """
            QFrame {
                border: 2px solid #800080;
                border-radius: 6px;
                background-color: #ffffff; /* was #fafafa -> pure white */
            }
        """
        )
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.output_stack)
        frame.setLayout(frame_layout)

        main_layout.addWidget(frame)
        self.setLayout(main_layout)

        self.load_benchmark_tabs()
        self.output_stack.setCurrentIndex(-1)
        self.current_thread = None

    def _pretty_name_for(self, file_path):
        base = os.path.splitext(os.path.basename(file_path))[0]
        return base.capitalize()

    def _button_color_for(self, fname):
        if "logic" in fname.lower():
            return "#8e44ad"
        elif any(x in fname.lower() for x in ["latency", "fps", "Inference"]):
            return "#27ae60"
        elif "validation" in fname.lower() or "accuracy" in fname.lower():
            return "#2980b9"
        else:
            return "#7f8c8d"

    def _reload_settings_dataset(self):
        """Reload dataset path from settings.json to reflect user changes."""
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r") as f:
                    cfg = json.load(f)
                self.dataset_path = (
                    cfg.get("dataset_path") or cfg.get("dataset") or self.dataset_path
                )
            except Exception as e:
                print(f"[WARN] Could not reload settings: {e}")

    def load_benchmark_tabs(self):
        """Dynamically create benchmark buttons and output panels."""
        if not os.path.isdir(self.benchmark_dir):
            print(f"[WARN] Benchmark dir not found: {self.benchmark_dir}")
            return

        all_scripts = []
        for dirpath, _, filenames in os.walk(self.benchmark_dir):
            for fname in filenames:
                if not fname.endswith(".py") or fname.startswith("__"):
                    continue
                if "logic" in fname.lower():
                    continue
                all_scripts.append(os.path.join(dirpath, fname))
        all_scripts.sort(key=lambda p: os.path.basename(p).lower())

        for file_path in all_scripts:
            fname = os.path.basename(file_path)
            tab_name = self._pretty_name_for(file_path)

            btn = QPushButton(tab_name)
            btn.setMinimumHeight(40)
            btn.clicked.connect(
                lambda _, n=tab_name, p=file_path: self.run_script(n, p)
            )

            color = self._button_color_for(fname)
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #555;
                }}
            """
            )

            # Disable FPS button only if it's truly FPS and dataset is LFW
            if "fps" in fname.lower() and "latency" not in fname.lower():
                ds_lower = (self.dataset_path or "").lower()
                if "lfw" in ds_lower:
                    btn.setEnabled(False)
                    btn.setToolTip(
                        "FPS benchmark only works with YTF (video) datasets."
                    )

            self.button_layout.addWidget(btn)

            # Matplotlib page
            page = QWidget()
            layout = QVBoxLayout()

            # >>> Pure white figure & canvas
            fig = Figure(figsize=(5, 4), facecolor="white")
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: white;")

            layout.addWidget(canvas)
            progress = QProgressBar()
            layout.addWidget(progress)
            page.setLayout(layout)

            idx = self.output_stack.addWidget(page)
            self.pages[tab_name] = (idx, fig, canvas, file_path, progress)

    def run_script(self, name, file_path):
        if name not in self.pages:
            return

        # 🔄 Refresh dataset in case user changed it in Settings
        self._reload_settings_dataset()

        idx, fig, canvas, file_path, progress = self.pages[name]
        self.output_stack.setCurrentIndex(idx)

        model_name = self.get_model_name() if callable(self.get_model_name) else None
        if not model_name:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "No model selected", ha="center", va="center", color="red"
            )
            canvas.draw()
            return

        if not self.dataset_path:
            QMessageBox.warning(
                self, "No Dataset", "Please select a dataset in Settings first."
            )
            return

        dataset_lower = (self.dataset_path or "").lower()
        iters = None
        extra_args = []

        # --- Detect dataset type ---
        is_video_dataset = any(
            x in dataset_lower for x in ["ytf", "aligned_images_db", "video"]
        )
        is_image_dataset = any(x in dataset_lower for x in ["lfw", "image", "photo"])

        # --- Video datasets (YTF etc.) ---
        if is_video_dataset:
            # --- VIDEO VALIDATION ACCURACY (YTF etc.) ---
            if "validation_accuracy" in os.path.basename(file_path).lower():

                # 1) Ask how many pairs to test
                num_pairs, ok = QInputDialog.getInt(
                    self,
                    "Number of Pairs",
                    "How many pairs to test?",
                    600,
                    100,
                    6000,
                    100,
                )
                if not ok:
                    return

                # 2) Metric type (ROC vs Confusion)
                dlg_metric = SelectMetricDialog(self)
                if dlg_metric.exec_() != QDialog.Accepted:
                    return

                if dlg_metric.selection == "roc":
                    file_path = os.path.join(
                        self.benchmark_dir,
                        "validation_accuracy",
                        "video",
                        "logic_roc_graph_video.py",
                    )
                else:
                    file_path = os.path.join(
                        self.benchmark_dir,
                        "validation_accuracy",
                        "video",
                        "logic_confusion_matrix_video.py",
                    )

                # 3) Person selection (start identity / ALL) at video root
                ds_for_dialog = self.dataset_path
                if os.path.isdir(os.path.join(ds_for_dialog, "aligned_images_DB")):
                    ds_for_dialog = os.path.join(ds_for_dialog, "aligned_images_DB")

                dlg = SelectSubjectsDialog(ds_for_dialog, self)
                dlg.list_widget.setSelectionMode(QListWidget.SingleSelection)

                if dlg.exec_() != QDialog.Accepted or not dlg.selected_subjects:
                    return

                if dlg.selected_subjects[0] == "__ALL__":
                    start_person = "__ALL__"
                    iters = -1  # ALL mode
                else:
                    start_person = dlg.selected_subjects[0]
                    iters = num_pairs

                extra_args = ["--start", start_person]

            # --- VIDEO LATENCY / FPS / OTHER ---
            else:
                ds_for_dialog = self.dataset_path
                if os.path.isdir(os.path.join(ds_for_dialog, "aligned_images_DB")):
                    ds_for_dialog = os.path.join(ds_for_dialog, "aligned_images_DB")

                dlg = SelectSubjectsDialog(ds_for_dialog, self)
                if dlg.exec_() != QDialog.Accepted or not dlg.selected_subjects:
                    return
                os.environ["YTF_SELECTED_SUBJECTS"] = ",".join(dlg.selected_subjects)

                num_runs, ok = QInputDialog.getInt(
                    self, "Number of Runs", "How many runs?", 2, 1, 100, 1
                )
                if not ok:
                    return
                os.environ["YTF_RUNS"] = str(num_runs)

        # --- Image datasets (LFW etc.) ---
        elif is_image_dataset:
            base_name = os.path.basename(file_path).lower()

            # Allow Latency & Accuracy, but block FPS (but don't block latency scripts)
            if "fps" in base_name and "latency" not in base_name:
                QMessageBox.warning(
                    self,
                    "FPS Unsupported",
                    "FPS benchmark is only supported for video datasets (e.g., YTF).",
                )
                return

            # Auto-fix: if user selected LFW parent folder, go one level deeper
            if os.path.isdir(os.path.join(self.dataset_path, "lfw-deepfunneled")):
                self.dataset_path = os.path.join(self.dataset_path, "lfw-deepfunneled")

            # ---------- SPECIAL CASE: LFW pair protocols (ROC + Confusion) ----------
            if "roc_lfw_pairs" in base_name or "confusion_lfw_pairs" in base_name:
                # Use all pairs from pairs.txt. No dialogs needed.
                lfw_root = self.dataset_path
                if os.path.basename(lfw_root).lower() == "lfw-deepfunneled":
                    lfw_root = os.path.dirname(lfw_root)

                pairs_file = os.path.join(lfw_root, "pairs.txt")
                if not os.path.isfile(pairs_file):
                    QMessageBox.warning(
                        self,
                        "pairs.txt not found",
                        f"Could not find pairs.txt next to the LFW folder:\n{pairs_file}",
                    )
                    return

                # These scripts don't accept --iters, so leave iters as None.
                extra_args = ["--pairs", pairs_file]

            # ---------- Generic image VALIDATION ACCURACY (logic_roc_graph / confusion) ----------
            elif "validation_accuracy" in base_name:
                # Step 1: Ask how many pairs to test
                num_pairs, ok = QInputDialog.getInt(
                    self,
                    "Number of Pairs",
                    "How many pairs to test?",
                    600,
                    100,
                    6000,
                    100,
                )
                if not ok:
                    return

                # Step 2: Ask which metric to run
                dlg_metric = SelectMetricDialog(self)
                if dlg_metric.exec_() != QDialog.Accepted:
                    return

                if dlg_metric.selection == "roc":
                    file_path = os.path.join(
                        self.benchmark_dir,
                        "validation_accuracy",
                        "image",
                        "logic_roc_graph.py",
                    )
                else:
                    file_path = os.path.join(
                        self.benchmark_dir,
                        "validation_accuracy",
                        "image",
                        "logic_confusion_matrix.py",
                    )

                ds_for_dialog = self.dataset_path
                if os.path.isdir(os.path.join(ds_for_dialog, "lfw-deepfunneled")):
                    ds_for_dialog = os.path.join(ds_for_dialog, "lfw-deepfunneled")

                dlg = SelectSubjectsDialog(ds_for_dialog, self)
                dlg.list_widget.setSelectionMode(QListWidget.SingleSelection)
                if dlg.exec_() != QDialog.Accepted or not dlg.selected_subjects:
                    return

                if dlg.selected_subjects[0] == "__ALL__":
                    start_person = "__ALL__"
                    iters = -1  # tell script to process ALL pairs
                else:
                    start_person = dlg.selected_subjects[0]
                    iters = num_pairs  # limited mode

                extra_args = ["--start", start_person]
                os.environ["LFW_START_PERSON"] = start_person
                os.environ["POS_RATIO"] = "0.5"

            # ---------- Latency / inference on image datasets ----------
            else:
                # Show person-selection dialog for latency, inference etc.
                people = sorted(
                    [
                        d
                        for d in os.listdir(self.dataset_path)
                        if os.path.isdir(os.path.join(self.dataset_path, d))
                    ]
                )
                if not people:
                    QMessageBox.warning(
                        self, "No Folders", "No people found in dataset path."
                    )
                    return

                dlg = SelectSubjectsDialog(self.dataset_path, self)
                if dlg.exec_() != QDialog.Accepted or not dlg.selected_subjects:
                    return

                selected_people = dlg.selected_subjects
                start_person = selected_people[0]
                os.environ["LFW_SELECTED_PEOPLE"] = ",".join(selected_people)

                img_count, ok2 = QInputDialog.getInt(
                    self, "Image Count", "How many images to include?", 10, 1, 10000, 1
                )
                if not ok2:
                    return

                num_runs, ok3 = QInputDialog.getInt(
                    self, "Number of Runs", "How many runs?", 2, 1, 100, 1
                )
                if not ok3:
                    return

                os.environ["LFW_START_PERSON"] = start_person
                os.environ["LFW_IMAGE_COUNT"] = str(img_count)
                os.environ["LFW_RUNS"] = str(num_runs)

        # --- Fallback: if unknown dataset ---
        else:
            QMessageBox.warning(
                self,
                "Unknown Dataset",
                "Could not determine dataset type — please use either LFW (images) or YTF (videos).",
            )
            return

        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.terminate()

        self.current_thread = RunnerThread(
            file_path,
            model_name,
            self.dataset_path,
            self.test_image,
            iters=iters,
            extra_args=extra_args,
        )
        self.current_thread.output_signal.connect(
            lambda msg: self.handle_output(msg, fig, canvas, progress)
        )
        self.current_thread.start()

    # ---------- Handle Output ----------

    def handle_output(self, msg, fig, canvas, progress):
        import numpy as np
        import matplotlib.cm as cm
        import os, json

        try:
            data = json.loads(msg)
        except Exception:
            print(f"[SCRIPT LOG] {msg}")
            return

        if not isinstance(data, dict):
            # Not a top-level JSON object (likely a line from the pretty block) → ignore
            print(f"[SCRIPT LOG] {msg}")
            return

        # ----- Progress -----
        if any(k in data for k in ["_type", "progress", "total"]) and (
            data.get("_type") == "progress" or ("progress" in data and "total" in data)
        ):
            try:
                cur = int(data.get("progress", 0))
                tot = int(data.get("total", 1))
                pct = int(100 * cur / tot)
                progress.setMaximum(100)
                progress.setValue(pct)

                # ✅ Include run info if present
                run_str = ""
                if "run" in data and "num_runs" in data:
                    run_str = f" | Run {data['run']}/{data['num_runs']}"

                # ✅ Auto-reset progress bar when new run starts
                if data.get("progress") == 1:
                    progress.reset()

                progress.setFormat(f"Processing {cur}/{tot} items ({pct}%)" + run_str)

            except Exception as e:
                print(f"[WARN] Progress parse failed: {e} | data={data}")
            return

        # ----- Logs -----
        if "log" in data:
            print(f"[SCRIPT LOG] {data['log']}")
            return

        # ----- Errors -----
        if "error" in data:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Error: {data['error']}",
                ha="center",
                va="center",
                color="red",
            )
            canvas.draw()
            return

        # ------------------------------------------------------------------
        # Auto-detect benchmark kind
        # ------------------------------------------------------------------
        kind = data.get("kind", "")
        fig.clear()
        ax = fig.add_subplot(111)

        # 1) Confusion Matrix
        if kind == "confusion_matrix":
            return handle_confusion(data, fig, canvas)

        # 2) Latency
        if kind == "latency" or "latency_series_all" in data:
            return handle_latency(data, fig, canvas, self)

        # 3) FPS
        if kind == "fps" or "fps_series_all" in data:
            return handle_fps(data, fig, canvas)

        # 4) ROC Image
        if kind == "roc_image":
            return handle_roc_image(data, fig, canvas)

        # 5) Image validation accuracy
        if kind == "accuracy_image":
            return handle_accuracy_image(data, fig, canvas)

        # 6) Video validation accuracy
        if kind == "accuracy_video":
            return handle_accuracy_video(data, fig, canvas)
