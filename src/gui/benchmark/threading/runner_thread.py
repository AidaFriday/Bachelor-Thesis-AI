import sys
import subprocess
import json
from PyQt5.QtCore import QThread, pyqtSignal


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
        self.extra_args = extra_args or []

    def run(self):
        try:
            cmd = [
                sys.executable,
                "-u",
                self.file_path,
                "--model",
                self.model_name,
            ]

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
