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
)
from PyQt5.QtCore import QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RunnerThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, file_path, model_name):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name

    def run(self):
        try:
            cmd = [sys.executable, self.file_path, "--model", self.model_name]
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


class BenchmarkPage(QWidget):
    def __init__(self, parent=None, get_model_name=None):
        super().__init__(parent)
        self.get_model_name = get_model_name

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.benchmark_dir = os.path.join(base_dir, "benchmark_parameters")

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
        # Button label is just the leaf filename (Accuracy, Fps, Latency, …)
        base = os.path.splitext(os.path.basename(file_path))[0]
        return base.capitalize()

    def load_benchmark_tabs(self):
        if not os.path.isdir(self.benchmark_dir):
            print(f"[WARN] Benchmark dir not found: {self.benchmark_dir}")
            return

        # Recurse through all subdirectories and pick up every .py file
        for dirpath, _, filenames in os.walk(self.benchmark_dir):
            for fname in sorted(f for f in filenames if f.endswith(".py")):
                file_path = os.path.join(dirpath, fname)
                tab_name = self._pretty_name_for(file_path)

                btn = QPushButton(tab_name)
                btn.setMinimumHeight(40)
                btn.clicked.connect(
                    lambda _, n=tab_name, p=file_path: self.run_script(n, p)
                )
                self.button_layout.addWidget(btn)

                page = QWidget()
                layout = QVBoxLayout()
                fig = Figure(figsize=(5, 4))
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                page.setLayout(layout)
                idx = self.output_stack.addWidget(page)

                self.pages[tab_name] = (idx, fig, canvas, file_path)

    def run_script(self, name, file_path):
        if name not in self.pages:
            return
        idx, fig, canvas, file_path = self.pages[name]
        self.output_stack.setCurrentIndex(idx)

        model_name = self.get_model_name() if self.get_model_name else None
        if not model_name:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "❌ No model selected",
                ha="center",
                va="center",
                color="red",
                fontsize=12,
            )
            canvas.draw()
            return

        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.terminate()

        self.current_thread = RunnerThread(file_path, model_name)
        self.current_thread.output_signal.connect(
            lambda msg: self.handle_output(msg, fig, canvas)
        )
        self.current_thread.start()

    def handle_output(self, msg, fig, canvas):
        # 1) parse a single JSON line from the worker
        try:
            data = json.loads(msg)
        except Exception:
            return

        fig.clear()
        ax = fig.add_subplot(111)

        # 2) explicit error from the worker
        if "error" in data:
            ax.text(
                0.5,
                0.5,
                f"Error: {data['error']}",
                ha="center",
                va="center",
                color="red",
                fontsize=12,
            )
            canvas.draw()
            return

        # 3) LATENCY view ---------------------------------------------------
        if "times" in data:
            times = list(data.get("times", []))
            dataset = data.get("dataset", "synthetic")
            model = data.get("model", "")
            ax.plot(
                range(1, len(times) + 1), times, marker="o", linestyle="-", color="blue"
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Latency (ms)")
            ax.set_title(f"Latency per Inference - {model} ({dataset})")
            ax.grid(True)

            if times:
                mean_ms = sum(times) / len(times)
                min_ms = min(times)
                max_ms = max(times)

                ax.axhline(mean_ms, linestyle="--", color="tab:blue", alpha=0.7)
                ax.text(
                    0.98,
                    0.98,
                    f"mean: {mean_ms:.2f} ms\nmin: {min_ms:.2f} ms  max: {max_ms:.2f} ms",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7, ec="gray"),
                )

            canvas.draw()
            return

        # 4) FPS view -------------------------------------------------------
        if "fps_series" in data:
            fps_series = list(data.get("fps_series", []))
            dataset = data.get("dataset", "synthetic")
            model = data.get("model", "")
            ax.plot(
                range(1, len(fps_series) + 1),
                fps_series,
                marker="o",
                linestyle="-",
                color="blue",
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("FPS")
            ax.set_title(f"Frames per Second - {model} ({dataset})")
            ax.grid(True)

            if fps_series:
                mean_fps = sum(fps_series) / len(fps_series)
                min_fps = min(fps_series)
                max_fps = max(fps_series)

                ax.axhline(mean_fps, linestyle="--", color="tab:blue", alpha=0.7)

                avg_ms = data.get("avg_ms")
                ms_line = (
                    f"\navg latency: {avg_ms:.2f} ms"
                    if isinstance(avg_ms, (int, float))
                    else ""
                )

                ax.text(
                    0.98,
                    0.98,
                    f"mean FPS: {mean_fps:.2f}\nmin: {min_fps:.2f}  max: {max_fps:.2f}{ms_line}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7, ec="gray"),
                )

            canvas.draw()
            return

        # 5) ACCURACY view --------------------------------------------------
        if "positives" in data and "negatives" in data:
            dataset = data.get("dataset", "synthetic")
            model = data.get("model", "")
            ax.hist(
                data["positives"],
                bins=50,
                alpha=0.6,
                label="Positive (user1)",
                color="g",
            )
            ax.hist(
                data["negatives"],
                bins=50,
                alpha=0.6,
                label="Negative (others)",
                color="r",
            )
            ax.axvline(
                data.get("threshold", 0.5),
                color="blue",
                linestyle="--",
                label=f"Threshold={data.get('threshold', 0.5)}",
            )
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.set_title(f"Face Verification - {model} ({dataset})")
            canvas.draw()
            return

        # 6) Fallback -------------------------------------------------------
        ax.text(
            0.5,
            0.5,
            "No plottable data received.",
            ha="center",
            va="center",
            fontsize=12,
        )
        canvas.draw()
