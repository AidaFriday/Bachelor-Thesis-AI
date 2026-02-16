import os
import json

from PyQt5.QtWidgets import QInputDialog, QMessageBox, QListWidget, QDialog

# Import dialogs
from gui.benchmark.dialogs.select_metric_dialog import SelectMetricDialog
from gui.benchmark.dialogs.select_subjects_dialog import SelectSubjectsDialog

# Import thread
from gui.benchmark.threading.runner_thread import RunnerThread


def run_script_logic(self, name, file_path):
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
        ax.text(0.5, 0.5, "No model selected", ha="center", va="center", color="red")
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
