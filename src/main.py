# ==== main.py ====

import sys
import os
from PyQt5.QtWidgets import QApplication
from windows.home.home_window import HomeWindow

# --- Auto-index all project files on startup ---
try:
    # Import the file indexer (stored under components/utilities)
    from components.utilities.file_indexer import update_file_index

    # Automatically rebuild file_index.json to reflect current structure
    update_file_index()
except Exception as e:
    # Fallback warning if the utilities module isn’t accessible
    print(f"[WARN] File indexing skipped due to error: {e}")


def main():
    """Launch the main PyQt5 application window."""
    app = QApplication(sys.argv)
    window = HomeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
