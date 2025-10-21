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
        self,
        file_path: str,
        model_name: str,
        dataset_path: str = None,
        test_image: str = None,
        iters: int = 50,
        extra_args=None,
        parent=None,
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.test_image = test_image
        self.iters = iters
        self.extra_args = extra_args or []

    def run(self):
        try:
            cmd = [
                sys.executable,
                self.file_path,
                "--model_path",
                self.model_name,
                "--dataset_path",
                self.dataset_path,
                "--iters",
                str(self.iters),
            ]
            if self.extra_args:
                cmd.extend(self.extra_args)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,  # only stdout -> GUI
                stderr=subprocess.DEVNULL,  # hide tqdm / logs
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
        self._printed_result_this_run = False

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
                QPushButton:hover {{ background-color: #555; }}
            """
            )

            if "fps" in fname.lower() and "latency" not in fname.lower():
                ds_lower = (self.dataset_path or "").lower()
                if "lfw" in ds_lower:
                    btn.setEnabled(False)
                    btn.setToolTip(
                        "FPS benchmark only works with YTF (video) datasets."
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

        is_video_dataset = any(
            x in dataset_lower for x in ["ytf", "aligned_images_db", "video"]
        )
        is_image_dataset = any(x in dataset_lower for x in ["lfw", "image", "photo"])

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

        elif is_image_dataset:
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

            if os.path.isdir(os.path.join(self.dataset_path, "lfw-deepfunneled")):
                self.dataset_path = os.path.join(self.dataset_path, "lfw-deepfunneled")

            if "validation_accuracy" in os.path.basename(file_path).lower():
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

                ds_for_dialog = self.dataset_path
                if os.path.isdir(os.path.join(ds_for_dialog, "lfw-deepfunneled")):
                    ds_for_dialog = os.path.join(ds_for_dialog, "lfw-deepfunneled")

                dlg = SelectSubjectsDialog(ds_for_dialog, self)
                dlg.list_widget.setSelectionMode(QListWidget.SingleSelection)
                if dlg.exec_() != QDialog.Accepted or not dlg.selected_subjects:
                    return
                start_person = dlg.selected_subjects[0]

                pos_ratio = 0.5
                os.environ["LFW_START_PERSON"] = start_person
                os.environ["POS_RATIO"] = str(pos_ratio)
                iters = num_pairs
            else:
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

        else:
            QMessageBox.warning(
                self, "Unknown Dataset", "Use either LFW (images) or YTF (videos)."
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
        self._printed_result_this_run = False
        self.current_thread.start()

    # ---------- Handle Output ----------
    def handle_output(self, msg, fig, canvas, progress):
        import numpy as np
        import os, json

        # 1) Try to parse JSON messages (the actual payloads come as JSON)
        try:
            data = json.loads(msg)
        except Exception:
            # ignore non-JSON noise completely
            return

        if not isinstance(data, dict):
            return

        # ignore plain log objects coming from the worker
        if "log" in data:
            return

        # 2) Plot VA results AND print the two console blocks once
        kind = data.get("kind", "")
        if kind == "accuracy_image":
            # --------- print the two blocks you want (once per run) ----------
            if not self._printed_result_this_run:
                # make a shallow copy and strip the huge ROC arrays for console output
                sanitized = dict(data)
                if "roc" in sanitized:
                    r = sanitized["roc"] or {}
                    sanitized["roc"] = {
                        "auc": r.get("auc"),
                        "points": (
                            len(r.get("fpr", []))
                            if isinstance(r.get("fpr", []), list)
                            else 0
                        ),
                    }

                pretty = json.dumps(sanitized, indent=2)
                print("[SCRIPT LOG] [RESULT]")
                for line in pretty.splitlines():
                    print(f"[SCRIPT LOG] {line}")

                # build the exact summary line you liked
                tp = int(data.get("tp", 0))
                fp = int(data.get("fp", 0))
                tn = int(data.get("tn", 0))
                fn = int(data.get("fn", 0))
                pos = int(data.get("pos_pairs", 0))
                neg = int(data.get("neg_pairs", 0))
                ids = int(data.get("unique_identities", 0))
                caps_pos = data.get("max_pos_per_identity", 0)
                caps_neg = data.get("max_neg_per_identity", 0)

                summary = (
                    f"[SUMMARY] TP:{tp} FP:{fp} TN:{tn} FN:{fn} | "
                    f"+:{pos} -:{neg} | IDs:{ids} | caps(+:{caps_pos}, -:{caps_neg})"
                )
                print(f"[SCRIPT LOG] {summary}")
                self._printed_result_this_run = True
            # -----------------------------------------------------------------

            # ---------- draw the ROC ----------
            fig.clear()
            ax = fig.add_subplot(111)

            model = data.get("model", "Unknown")
            start_person = data.get("start_person") or "N/A"
            ax.set_title(
                f"Model: {model} – Validation Accuracy (Image) – Start: {start_person}"
            )

            roc = data.get("roc", {})
            fpr = np.array(roc.get("fpr", []), dtype=float)
            tpr = np.array(roc.get("tpr", []), dtype=float)
            auc = float(roc.get("auc", float("nan")))
            tp = int(data.get("tp", 0))
            fp = int(data.get("fp", 0))
            tn = int(data.get("tn", 0))
            fn = int(data.get("fn", 0))
            thr = float(data.get("threshold", 0))
            acc = float(data.get("accuracy", 0))
            num_pairs = int(data.get("num_pairs", 0))
            elapsed = float(data.get("elapsed_sec", 0))

            if fpr.size and tpr.size:
                ax.plot(fpr, tpr, linewidth=1.8)
                ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.grid(True, alpha=0.25)

                P = tp + fn
                N = tn + fp
                if P > 0 and N > 0:
                    op_tpr = tp / P
                    op_fpr = fp / N
                    ax.scatter([op_fpr], [op_tpr], s=28)
                    ax.annotate(
                        "thr",
                        (op_fpr, op_tpr),
                        fontsize=8,
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

            # metrics box (top-right)
            metrics_text = (
                f"AUC: {auc:.3f}\n"
                f"Accuracy: {acc*100:.2f}%\n"
                f"Threshold: {thr:.3f}\n"
                f"Pairs: {num_pairs}   Time: {elapsed:.1f}s\n"
                f"+/–: {data.get('pos_pairs',0)}/{data.get('neg_pairs',0)}\n"
                f"IDs: {data.get('unique_identities',0)}   "
                f"caps(+:{data.get('max_pos_per_identity',0)}, -:{data.get('max_neg_per_identity',0)})"
            )
            ax.text(
                0.98,
                0.02,
                metrics_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75"),
            )

            # tiny confusion matrix (bottom-left)
            conf_text = f"TP:{tp}  FP:{fp}\nTN:{tn}  FN:{fn}"
            ax.text(
                0.02,
                0.02,
                conf_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75"),
            )

            fig.tight_layout()
            canvas.draw()
            return

        # for other kinds (fps/latency), do nothing special here

        print("[WARN] Unrecognized data keys:", data.keys())
