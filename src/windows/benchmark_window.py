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
        iters=50,
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
                "-u",  # <--- add this for unbuffered stdout/stderr
                self.file_path,
                "--model",
                self.model_name,
                "--iters",
                str(self.iters),
            ]

            if self.dataset_path:
                cmd.extend(["--dataset", self.dataset_path])

            # append any extra CLI arguments (e.g., --start-person, --pos-ratio)
            if self.extra_args:
                cmd.extend(self.extra_args)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",  # ✅ force UTF-8 decoding
                errors="replace",  # ✅ avoid crash on unsupported chars
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
        iters = 0
        extra_args = []

        # --- Detect dataset type ---
        is_video_dataset = any(
            x in dataset_lower for x in ["ytf", "aligned_images_db", "video"]
        )
        is_image_dataset = any(x in dataset_lower for x in ["lfw", "image", "photo"])

        # --- Video datasets (YTF etc.) ---
        if is_video_dataset:
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
            # Allow Latency & Accuracy, but block FPS (but don't block latency scripts)
            if (
                "fps" in os.path.basename(file_path).lower()
                and "latency" not in os.path.basename(file_path).lower()
            ):
                QMessageBox.warning(
                    self,
                    "FPS Unsupported",
                    "FPS benchmark is only supported for video datasets (e.g., YTF).",
                )
                return

            # Auto-fix: if user selected LFW parent folder, go one level deeper
            if os.path.isdir(os.path.join(self.dataset_path, "lfw-deepfunneled")):
                self.dataset_path = os.path.join(self.dataset_path, "lfw-deepfunneled")

            # For validation accuracy, skip person selection — script handles pairs internally
            if "validation_accuracy" in os.path.basename(file_path).lower():

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

                # ✅ Step 3: Ask which metric to run
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

                # extra_args stays [] for VA

                ds_for_dialog = self.dataset_path
                if os.path.isdir(os.path.join(ds_for_dialog, "lfw-deepfunneled")):
                    ds_for_dialog = os.path.join(ds_for_dialog, "lfw-deepfunneled")

                dlg = SelectSubjectsDialog(ds_for_dialog, self)
                dlg.list_widget.setSelectionMode(QListWidget.SingleSelection)
                if dlg.exec_() != QDialog.Accepted or not dlg.selected_subjects:
                    return

                if dlg.selected_subjects[0] == "__ALL__":
                    start_person = "__ALL__"
                    iters = -1  # ✅ tell script to process ALL pairs
                else:
                    start_person = dlg.selected_subjects[0]
                    iters = num_pairs  # ✅ normal limited mode

                extra_args = ["--start", start_person]

                # pass for ROC / Confusion scripts
                os.environ["LFW_START_PERSON"] = start_person
                os.environ["POS_RATIO"] = "0.5"

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

                progress.setFormat(f"Processing {cur}/{tot} frames ({pct}%)" + run_str)

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

        # ===================== CONFUSION MATRIX =====================
        if data.get("kind") == "confusion_matrix":
            tp, tn, fp, fn = data["tp"], data["tn"], data["fp"], data["fn"]
            threshold = data["threshold"]
            model = data.get("model", "")
            dataset = data.get("dataset", "")

            fig.clear()
            ax = fig.add_subplot(111)
            cm = np.array([[tp, fp], [fn, tn]], dtype=float)
            im = ax.imshow(cm, cmap="Blues")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred POS", "Pred NEG"])
            ax.set_yticklabels(["Actual POS", "Actual NEG"])

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        int(cm[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=12,
                    )

            ax.set_title(
                f"Confusion Matrix - {model} on {dataset}\nThr={threshold:.4f}"
            )

            # Draw confusion matrix colors
            fig.colorbar(im, ax=ax)

            # ✅ Shift matrix left to make room for metrics box
            fig.subplots_adjust(right=0.75)

            # ✅ Build metrics text (reads clean and matches terminology)
            metrics = (
                f"Accuracy: {data['accuracy']*100:.2f}%\n"
                f"Precision: {data['precision']*100:.2f}%\n"
                f"Recall (TAR): {data['recall']*100:.2f}%\n"
                f"Specificity (TNR): {data['specificity']*100:.2f}%\n"
                f"F1 Score: {data['f1']*100:.2f}%\n"
                f"FAR: {data['far']*100:.2f}%\n"
                f"FRR: {data['frr']*100:.2f}%"
            )

            # ✅ Place metrics box OUTSIDE the plot area properly
            fig.text(
                1.15,
                0.5,
                metrics,
                transform=ax.transAxes,
                fontsize=10,
                va="center",
                ha="left",
                color="white",  # ← text color for dark background
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    fc="#222222",  # ← dark background
                    ec="#555555",  # ← subtle border
                    alpha=0.9,  # ← slightly transparent
                ),
            )

            canvas.draw()
            return

        # ===================== LATENCY MODE =====================

        if kind == "latency" or "latency_series_all" in data:
            latency_series_all = data.get("latency_series_all", [])
            run_avgs = data.get("runs", [])
            dataset = data.get("dataset", "")
            model = data.get("model", "")

            num_runs = len(latency_series_all)
            cmap = cm.get_cmap("tab10", max(1, num_runs))
            styles = ["-", "--", "-.", ":"]
            colors = [cmap(i / max(1, num_runs - 1)) for i in range(num_runs)]

            # ✅ Always use top-level latency_reports directory
            base_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "latency_reports",
            )
            os.makedirs(base_dir, exist_ok=True)

            # ✅ Include the originating file name if present in payload
            source_file = data.get("source_file", "unknown")
            all_run_data = {"source_file": source_file, "runs": []}

            problem_files = []

            for i, latencies in enumerate(latency_series_all):
                if not latencies:
                    problem_files.append(f"Run_{i+1}_empty")
                    continue

                latencies = np.array(latencies)
                avg_ms = float(np.mean(latencies))
                std_ms = float(np.std(latencies))
                min_idx = int(np.argmin(latencies))
                max_idx = int(np.argmax(latencies))
                min_ms = float(latencies[min_idx])
                max_ms = float(latencies[max_idx])

                # Try to attach filenames if available
                frame_paths = data.get("frame_paths_all", [])
                run_paths = frame_paths[i] if i < len(frame_paths) else []
                min_file = (
                    os.path.basename(run_paths[min_idx])
                    if run_paths and min_idx < len(run_paths)
                    else f"frame_{min_idx+1}"
                )
                max_file = (
                    os.path.basename(run_paths[max_idx])
                    if run_paths and max_idx < len(run_paths)
                    else f"frame_{max_idx+1}"
                )

                run_entry = {
                    "run": i + 1,
                    "min_ms": min_ms,
                    "max_ms": max_ms,
                    "avg_ms": avg_ms,
                    "min_file": min_file,
                    "max_file": max_file,
                    "p50_ms": float(np.percentile(latencies, 50)),
                    "p90_ms": float(np.percentile(latencies, 90)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "p99_ms": float(np.percentile(latencies, 99)),
                    "std_ms": std_ms,
                }
                all_run_data["runs"].append(run_entry)

                # Plot per run
                ax.plot(
                    range(1, len(latencies) + 1),
                    latencies,
                    linestyle=styles[i % len(styles)],
                    color=colors[i],
                    linewidth=1.5,
                    label=f"Run {i+1} – {avg_ms:.2f} ms",
                )

            # ✅ Save all runs into one JSON file
            report_path = os.path.join(base_dir, "latency_report.json")
            with open(report_path, "w") as f:
                json.dump(all_run_data, f, indent=4)

            if problem_files:
                with open(os.path.join(base_dir, "problem_runs.txt"), "w") as f:
                    f.write("\n".join(problem_files))

            # Place legend outside the plot area, aligned to the right
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
                title=(
                    "Latency per Run"
                    if kind == "latency" or "latency_series_all" in data
                    else "Per-Run Averages"
                ),
                fontsize=9,
            )
            fig.subplots_adjust(right=0.8)

            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Latency (ms)")

            # --- format dataset/model names more nicely ---
            dataset_name = os.path.basename(dataset).replace("-", " ").title()
            if "lfw" in dataset_name.lower():
                dataset_name = "LFW (Deepfunneled)"
            if "ytf" in dataset_name.lower():
                dataset_name = "YTF (Aligned)"
            if not model:
                model = "Unknown Model"

            ax.set_title(f"Per-Frame Latency – {model} on {dataset_name}")

            ax.grid(True)
            canvas.draw()
            return

        # ===================== FPS MODE =====================
        if kind == "fps" or "fps_series_all" in data:
            fps_series_all = data.get("fps_series_all", [])
            run_avgs = data.get("runs", [])
            dataset = data.get("dataset", "")
            model = data.get("model", "")

            num_runs = len(fps_series_all)
            cmap = cm.get_cmap("tab10", max(1, num_runs))
            styles = ["-", "--", "-.", ":"]
            colors = [cmap(i / max(1, num_runs - 1)) for i in range(num_runs)]

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

            # Place legend outside the plot area, aligned to the right
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
                title=(
                    "Latency per Run"
                    if kind == "latency" or "latency_series_all" in data
                    else "Per-Run Averages"
                ),
                fontsize=9,
            )
            fig.subplots_adjust(right=0.8)

            ax.set_xlabel("Iteration")
            ax.set_ylabel("FPS")
            ax.set_title(f"Frames per Second – {model} ({dataset})")
            ax.grid(True)
            # --- compute & annotate overall average FPS ---
            if run_avgs:
                avg_all = float(np.mean(run_avgs))
                num_runs = len(run_avgs)
                ax.text(
                    1.02,  # ← moved left so it stays visible
                    0.35,
                    f"Average FPS over {num_runs} run(s): {avg_all:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="gray",
                        boxstyle="round,pad=0.3",
                        alpha=0.7,
                    ),
                )

            canvas.draw()
            return

        # ===================== ROC IMAGE RESULT =====================
        if data.get("kind") == "roc_image":
            img_path = data.get("path")
            if not os.path.exists(img_path):
                print(f"[WARN] ROC image not found at {img_path}")
                return

            import matplotlib.image as mpimg

            fig.clear()
            fig.set_facecolor("white")
            ax = fig.add_axes([0, 0, 1, 1], frameon=False, facecolor="white")
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_axis_off()

            # ---- NEW METRIC OVERLAY ----
            auc_val = data.get("auc")
            eer = data.get("eer")
            tar = data.get("tar_far_1e3")
            pairs = data.get("pairs_tested")
            model = data.get("model")
            dataset = data.get("dataset")

            # Build annotation text
            lines = []
            if model and dataset:
                lines.append(f"{model} on {dataset}")
            if auc_val is not None:
                lines.append(f"AUC: {float(auc_val):.4f}")
            if eer is not None:
                lines.append(f"EER: {float(eer)*100:.2f}%")
            if tar is not None:
                lines.append(f"TAR@FAR=1e-3: {float(tar)*100:.2f}%")
            if pairs:
                lines.append(f"Pairs: {int(pairs)}")

            best_thr = data.get("best_threshold")
            if best_thr is not None:
                lines.insert(3, f"Best Thr (Youden J): {float(best_thr):.4f}")

            # Draw overlay (bottom-right)
            if lines:
                fig.text(
                    0.98,
                    0.04,
                    "\n".join(lines),
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9
                    ),
                )

            canvas.setStyleSheet("background-color: white;")
            canvas.draw()
            return

            # ===================== SIMPLE IMAGE ACCURACY =====================
        if data.get("kind") == "accuracy_image_simple":
            model = data.get("model", "Unknown")
            dataset = data.get("dataset", "Unknown")
            acc = float(data.get("accuracy", 0)) * 100.0
            threshold = float(data.get("threshold", 0))
            pairs = int(data.get("pairs_tested", 0))

            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")

            text = (
                f"Model: {model}\n"
                f"Dataset: {dataset}\n\n"
                f"Accuracy: {acc:.2f}%\n"
                f"Best Threshold: {threshold:.4f}\n"
                f"Pairs Tested: {pairs}"
            )

            ax.text(
                0.5,
                0.5,
                text,
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8),
            )

            canvas.draw()
            return

        # ===================== VALIDATION ACCURACY (IMAGE) =====================
        if kind == "accuracy_image":
            model = data.get("model", "Unknown")
            dataset = data.get("dataset", "Unknown")
            acc_pct = float(data.get("accuracy", 0)) * 100.0
            threshold = float(data.get("threshold", 0))
            elapsed = float(data.get("elapsed_sec", 0))
            start_person = data.get("start_person") or "N/A"
            tp = int(data.get("tp", 0))
            fp = int(data.get("fp", 0))
            tn = int(data.get("tn", 0))
            fn = int(data.get("fn", 0))

            # 👉 If we have a ROC image, render it directly (full-bleed, white)
            roc_png = data.get("roc_png")
            if roc_png and os.path.isfile(roc_png):
                import matplotlib.image as mpimg

                img = mpimg.imread(roc_png)

                fig.clear()
                fig.set_facecolor("white")
                ax = fig.add_axes([0, 0, 1, 1], frameon=False, facecolor="white")
                ax.imshow(img)
                # overlay the fixed-threshold stats on top of the ROC PNG (bottom-right)
                t_fixed = data.get("threshold")
                tpr_fixed = data.get("tpr_at_fixed")
                fpr_fixed = data.get("fpr_at_fixed")

                if (
                    t_fixed is not None
                    and tpr_fixed is not None
                    and fpr_fixed is not None
                ):
                    fig.text(
                        0.98,
                        0.04,  # move if it overlaps your progress bar
                        f"TAR@\u03c4={t_fixed:.3f}: {tpr_fixed:.3f}\n"
                        f"FAR@\u03c4={t_fixed:.3f}: {fpr_fixed:.3f}",
                        ha="right",
                        va="bottom",
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9
                        ),
                        zorder=20,
                    )

                ax.set_axis_off()
                for s in ax.spines.values():
                    s.set_visible(False)
                canvas.setStyleSheet("background-color: white;")
                canvas.draw()
                return

            # (fallback to existing text summary view)
            ax.set_title(
                f"Model: {model} – Validation Accuracy (Image) – Start: {start_person}"
            )
            ax.text(
                1.05,
                0.85,
                "Confusion Matrix\n"
                f"True Positive : {tp}\n"
                f"False Positive: {fp}\n"
                f"True Negative : {tn}\n"
                f"False Negative: {fn}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75"),
            )
            ax.text(
                1.05,
                0.55,
                f"Accuracy: {acc_pct:.2f}%\nThreshold: {threshold:.3f}\nElapsed: {elapsed:.2f}s",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                color="black",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.75", alpha=0.9),
            )
            fig.subplots_adjust(right=0.78)
            ax.set_ylabel("Value")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            canvas.draw()
            return

        # ===================== VALIDATION ACCURACY (VIDEO) =====================
        elif kind == "accuracy_video":
            # Try to display ROC if available (full-bleed, white)
            roc_png = data.get("roc_png")
            if roc_png and os.path.isfile(roc_png):
                import matplotlib.image as mpimg

                img = mpimg.imread(roc_png)

                fig.clear()
                fig.set_facecolor("white")
                ax = fig.add_axes([0, 0, 1, 1], frameon=False, facecolor="white")
                ax.imshow(img)
                # overlay the fixed-threshold stats on top of the ROC PNG (bottom-right)

                t_fixed = data.get("threshold")
                tpr_fixed = data.get("tpr_fixed", data.get("tpr_at_fixed"))
                fpr_fixed = data.get("fpr_fixed", data.get("fpr_at_fixed"))

                if (
                    t_fixed is not None
                    and tpr_fixed is not None
                    and fpr_fixed is not None
                ):
                    fig.text(
                        0.98,
                        0.04,  # move if it overlaps your progress bar
                        f"TAR@\u03c4={t_fixed:.3f}: {tpr_fixed:.3f}\n"
                        f"FAR@\u03c4={t_fixed:.3f}: {fpr_fixed:.3f}",
                        ha="right",
                        va="bottom",
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9
                        ),
                        zorder=20,
                    )

                ax.set_axis_off()
                for s in ax.spines.values():
                    s.set_visible(False)
                canvas.setStyleSheet("background-color: white;")
                canvas.draw()
                return

            # Fallback: simple text if no ROC image is present
            model = data.get("model", "Unknown")
            dataset = data.get("dataset", "Unknown")
            acc_pct = float(data.get("accuracy", 0)) * 100.0
            auc = data.get("auc")
            eer = data.get("eer")
            ax.set_title(f"Model: {model} – Validation Accuracy (Video)")
            lines = [f"Accuracy: {acc_pct:.2f}%"]
            if auc is not None:
                lines.append(f"AUC: {auc:.4f}")
            if eer is not None:
                lines.append(f"EER: {float(eer)*100:.2f}%")
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.75", alpha=0.9),
            )
            ax.axis("off")
            canvas.draw()
            return

        # ------------------ fallback ------------------
        print("[WARN] Unrecognized data:", data.keys())
