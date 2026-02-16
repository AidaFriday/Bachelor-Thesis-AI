# ==== main.py (inside src/core/) ====

import sys
import os
import atexit
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication
from gui.home.home_window import HomeWindow

try:
    from components.utilities.file_indexer import update_file_index

    update_file_index()
except Exception as e:
    print(f"[WARN] File indexing skipped due to error: {e}")


def delete_pycache():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("[CLEANUP] Removing __pycache__ folders...")

    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            if d == "__pycache__":
                full = os.path.join(dirpath, d)
                try:
                    shutil.rmtree(full)
                    print(f"[CLEANUP] Removed: {full}")
                except Exception as e:
                    print(f"[CLEANUP] Failed to remove {full}: {e}")


atexit.register(delete_pycache)


def main():
    app = QApplication(sys.argv)
    window = HomeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
