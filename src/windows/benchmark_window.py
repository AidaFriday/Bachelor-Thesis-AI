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


# ------------------ New Dialog ------------------
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
                item.setCheckState(0)  # Unchecked
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

            # Always include dataset if available
            if self.dataset_path:
                cmd.extend(["--dataset", self.dataset_path])

            # validation_accuracy only uses test image
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


# ------------------ Main GUI Page ------------------
class BenchmarkPage(QWidget):
    def __init__(self, parent=None, get_model_name=None):
        super().__init__(parent)
        self.get_model_name = get_model_name

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.benchmark_dir = os.path.join(base_dir, "benchmark_parameters")

        # fixed test image path
        self.test_image = os.path.join(
            self.benchmark_dir, "validation_accuracy", "test_image", "test_image1.jpg"
        )

        # Load dataset from settings.json
        settings_path = os.path.join(base_dir, "settings.json")
        if os.path.exists(settings_path):
            with open(settings_path, "r") as f:
                cfg = json.load(f)
            self.dataset_path = cfg.get(
                "dataset", r"C:\programming\Datasets\LFW\lfw-deepfunneled"
            )
        else:
            self.dataset_path = r"C:\programming\Datasets\LFW\lfw-deepfunneled"

        main_layout = QVBoxLayout()

        # --- row of buttons ---
        self.button_layout = QHBoxLayout()
        main_layout.addLayout(self.button_layout)

        # --- output area ---
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

    def _pretty_name_for(self, file_path: str) -> str:
        base = os.path.splitext(os.path.basename(file_path))[0]
        return base.capitalize()

    def _button_color_for(self, fname: str) -> str:
        if "logic" in fname.lower():
            return "#8e44ad"
        elif (
            "latency" in fname.lower()
            or "fps" in fname.lower()
            or "inference" in fname.lower()
        ):
            return "#27ae60"
        elif "validation" in fname.lower() or "accuracy" in fname.lower():
            return "#2980b9"
        else:
            return "#7f8c8d"

    def load_benchmark_tabs(self):
        if not os.path.isdir(self.benchmark_dir):
            print(f"[WARN] Benchmark dir not found: {self.benchmark_dir}")
            return

        all_scripts = []
        for dirpath, _, filenames in os.walk(self.benchmark_dir):
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                if fname.startswith("__") or fname.startswith("."):
                    continue
                if "logic" in fname.lower():
                    continue
                all_scripts.append(os.path.join(dirpath, fname))

        all_scripts.sort(key=lambda p: os.path.basename(p).lower())

        for file_path in all_scripts:
            fname = os.path.basename(file_path)
            tab_name = self._pretty_name_for(file_path)
            if tab_name in self.pages:
                tab_name += f"_{len(self.pages)}"

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

            self.button_layout.addWidget(btn)

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

        idx, fig, canvas, file_path, progress = self.pages[name]
        self.output_stack.setCurrentIndex(idx)

        model_name = self.get_model_name() if self.get_model_name else None
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
        iters = 50  # default always defined

        if "ytf" in dataset_lower or "aligned_images_db" in dataset_lower:
            dataset_name = "YTF (aligned)"
            dlg = SelectSubjectsDialog(self.dataset_path, self)
            if dlg.exec_() != QDialog.Accepted or not dlg.selected_subjects:
                return

            selected_subjects = dlg.selected_subjects
            os.environ["YTF_SELECTED_SUBJECTS"] = ",".join(selected_subjects)
            print(
                f"[INFO] Selected {len(selected_subjects)} people: {', '.join(selected_subjects[:5])}"
            )

            num_runs, ok = QInputDialog.getInt(
                self,
                "Number of Runs",
                "How many runs do you want to perform?",
                5,
                1,
                100,
                1,
            )
            if not ok:
                return
            os.environ["YTF_RUNS"] = str(num_runs)

            # no frame popup
            iters = 50
        else:
            dataset_name = (
                "LFW"
                if "lfw" in dataset_lower
                else os.path.basename(self.dataset_path) or "synthetic"
            )

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

        if "fps_series_all" in data:
            fps_series_all = data.get("fps_series_all", [])
            dataset = data.get("dataset", "synthetic")
            model = data.get("model", "")

            # Pick color palette for multiple runs
            import matplotlib.cm as cm
            import numpy as np

            colors = cm.get_cmap("tab10", len(fps_series_all))

        if "fps_series_all" in data:
            fps_series_all = data.get("fps_series_all", [])
            run_avgs = data.get("runs", [])
            dataset = data.get("dataset", "synthetic")
            model = data.get("model", "")

            import matplotlib.cm as cm
            import numpy as np

            num_runs = len(fps_series_all)

            # ✅ Choose colormap dynamically depending on number of runs
            if num_runs <= 10:
                cmap = cm.get_cmap("tab10", num_runs)
            elif num_runs <= 20:
                cmap = cm.get_cmap("tab20", num_runs)
            else:
                # fallback: continuous hue gradient for large N
                cmap = cm.get_cmap("hsv", num_runs)

            # Generate distinct colors evenly spaced
            colors = [cmap(i / max(1, num_runs - 1)) for i in range(num_runs)]

            # Optional: alternate line styles for clarity if many runs
            line_styles = ["-", "--", "-.", ":"]
            ax.set_prop_cycle(None)  # reset color cycle

            for i, fps_series in enumerate(fps_series_all):
                avg_fps = run_avgs[i] if i < len(run_avgs) else np.mean(fps_series)
                ax.plot(
                    range(1, len(fps_series) + 1),
                    fps_series,
                    linestyle=line_styles[i % len(line_styles)],
                    color=colors[i],
                    linewidth=1.5,
                    label=f"Run {i+1}  ({avg_fps:.2f} FPS)",  # ✅ show averages in legend
                )

            ax.legend(loc="upper right", title="Per-Run Averages", fontsize=9)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("FPS")
            ax.set_title(f"Frames per Second - {model} ({dataset})")
            ax.grid(True)
            canvas.draw()
