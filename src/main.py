# ==== main.py ====

import sys
from PyQt5.QtWidgets import QApplication
from windows.home_window import HomeWindow


def main():
    app = QApplication(sys.argv)
    window = HomeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
