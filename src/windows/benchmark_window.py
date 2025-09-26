import os
import sys
import subprocess
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QStackedWidget, QFrame, QTextEdit
)
from PyQt5.QtCore import QThread, pyqtSignal


class RunnerThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, file_path, model_name):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name

    def run(self):
        try:
            cmd = [sys.executable, self.file_path, "--model", self.model_name]
            self.output_signal.emit(f"[DEBUG] Running command: {' '.join(cmd)}\n")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                self.output_signal.emit(line.rstrip())
            process.stdout.close()
            process.wait()
            self.output_signal.emit(f"\n[INFO] Finished {os.path.basename(self.file_path)}\n")
        except Exception as e:
            self.output_signal.emit(f"[ERROR] Failed to run {self.file_path}: {e}\n")


class BenchmarkPage(QWidget):
    def __init__(self, parent=None, get_model_name=None):
        """
        get_model_name: callback returning the currently selected model
        """
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
        frame.setStyleSheet("""
            QFrame {
                border: 2px solid #800080;
                border-radius: 6px;
                background-color: #fafafa;
            }
        """)
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.output_stack)
        frame.setLayout(frame_layout)

        main_layout.addWidget(frame)
        self.setLayout(main_layout)

        self.load_benchmark_tabs()
        self.output_stack.setCurrentIndex(-1)
        self.current_thread = None

    def load_benchmark_tabs(self):
        if not os.path.isdir(self.benchmark_dir):
            print(f"[WARN] Benchmark dir not found: {self.benchmark_dir}")
            return

        for fname in sorted(os.listdir(self.benchmark_dir)):
            if not fname.endswith(".py"):
                continue
            file_path = os.path.join(self.benchmark_dir, fname)
            tab_name = os.path.splitext(fname)[0]

            # button
            btn = QPushButton(tab_name.capitalize())
            btn.setMinimumHeight(40)
            btn.clicked.connect(lambda _, n=tab_name, p=file_path: self.run_script(n, p))
            self.button_layout.addWidget(btn)

            # output page
            page = QWidget()
            layout = QVBoxLayout()
            text_box = QTextEdit()
            text_box.setReadOnly(True)
            layout.addWidget(text_box)
            page.setLayout(layout)
            idx = self.output_stack.addWidget(page)

            self.pages[tab_name] = (idx, text_box)

    def run_script(self, name, file_path):
        if name not in self.pages:
            return
        idx, text_box = self.pages[name]
        self.output_stack.setCurrentIndex(idx)
        text_box.clear()

        model_name = self.get_model_name() if self.get_model_name else None
        if not model_name:
            msg = "[ERROR] No model selected in settings."
            text_box.setPlainText(msg)
            return

        text_box.append(f"[DEBUG] Selected model: {model_name}\n▶ Running {os.path.basename(file_path)}...\n")

        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.terminate()

        self.current_thread = RunnerThread(file_path, model_name)
        self.current_thread.output_signal.connect(lambda msg: text_box.append(msg))
        self.current_thread.start()
