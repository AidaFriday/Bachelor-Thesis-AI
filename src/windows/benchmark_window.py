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
)
from PyQt5.QtCore import QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
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


# ------------------ Worker Thread ------------------
class RunnerThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(
        self, file_path, model_name, dataset_path=None, test_image=None, iters=50
    ):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.test_image = test_image
        self.iters = iters

    def run(self):
        try:
            cmd = [
                sys.executable,
                self.file_path,
                "--model",
                self.model_name,
                "--iters",
                str(self.iters),
            ]

            if self.dataset_path:
                cmd.extend(["--dataset", self.dataset_path])

            if (
                "validation_accuracy" in os.path.basename(self.file_path)
                and self.test_image
            ):
                cmd.extend(["--test-image", self.test_image])

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
                background-color: #fafafa;
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
        elif any(x in fname.lower() for x in ["latency", "fps", "inference"]):
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

            # Disable FPS button if dataset currently LFW
            if "fps" in fname.lower():
                ds_lower = (self.dataset_path or "").lower()
                if "lfw" in ds_lower:
                    btn.setEnabled(False)
                    btn.setToolTip(
                        "FPS benchmark only works with YTF (video) datasets)."
                    )

            self.button_layout.addWidget(btn)

            # Matplotlib page
            page = QWidget()
            layout = QVBoxLayout()
            fig = Figure(figsize=(5, 4))
            canvas = FigureCanvas(fig)
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
        iters = 50

        # If YTF, open subject picker correctly
        if "ytf" in dataset_lower or "aligned_images_db" in dataset_lower:
            ds_for_dialog = self.dataset_path
            if os.path.isdir(os.path.join(ds_for_dialog, "aligned_images_DB")):
                ds_for_dialog = os.path.join(ds_for_dialog, "aligned_images_DB")

            dlg = SelectSubjectsDialog(ds_for_dialog, self)
            if dlg.exec_() != QDialog.Accepted or not dlg.selected_subjects:
                return
            os.environ["YTF_SELECTED_SUBJECTS"] = ",".join(dlg.selected_subjects)
            num_runs, ok = QInputDialog.getInt(
                self, "Number of Runs", "How many runs?", 5, 1, 100, 1
            )
            if not ok:
                return
            os.environ["YTF_RUNS"] = str(num_runs)
        else:
            if "fps" in os.path.basename(file_path).lower():
                QMessageBox.warning(
                    self,
                    "FPS Unsupported",
                    "FPS benchmark is only supported for YTF (video) datasets.",
                )
                return

        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.terminate()

        self.current_thread = RunnerThread(
            file_path, model_name, self.dataset_path, self.test_image, iters=iters
        )
        self.current_thread.output_signal.connect(
            lambda msg: self.handle_output(msg, fig, canvas, progress)
        )
        self.current_thread.start()

    # ---------- Handle Output ----------
    def handle_output(self, msg, fig, canvas, progress):
        try:
            data = json.loads(msg)
        except Exception:
            print(f"[SCRIPT LOG] {msg}")
            return

        if "progress" in data and "total" in data:
            pct = int(100 * data["progress"] / data["total"])
            progress.setValue(pct)
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"Processing {data['progress']}/{data['total']} images\n({pct}%)",
                ha="center",
                va="center",
                fontsize=12,
                color="blue",
            )
            canvas.draw()
            return

        if "log" in data:
            print(f"[SCRIPT LOG] {data['log']}")
            return

        fig.clear()
        ax = fig.add_subplot(111)

        if "error" in data:
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

        # Plot FPS results
        if "fps_series_all" in data:
            fps_series_all = data.get("fps_series_all", [])
            run_avgs = data.get("runs", [])
            dataset = data.get("dataset", "synthetic")
            model = data.get("model", "")
            import matplotlib.cm as cm
            import numpy as np

            num_runs = len(fps_series_all)
            cmap = cm.get_cmap("tab10", max(1, num_runs))
            colors = [cmap(i / max(1, num_runs - 1)) for i in range(num_runs)]
            styles = ["-", "--", "-.", ":"]

            for i, fps_series in enumerate(fps_series_all):
                avg_fps = run_avgs[i] if i < len(run_avgs) else np.mean(fps_series)
                ax.plot(
                    range(1, len(fps_series) + 1),
                    fps_series,
                    linestyle=styles[i % len(styles)],
                    color=colors[i],
                    linewidth=1.5,
                    label=f"Run {i+1} ({avg_fps:.2f} FPS)",
                )

            ax.legend(loc="upper right", title="Per-Run Averages", fontsize=9)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("FPS")
            ax.set_title(f"Frames per Second - {model} ({dataset})")
            ax.grid(True)
            canvas.draw()
