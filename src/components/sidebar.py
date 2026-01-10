# components/sidebar.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QPropertyAnimation


class SideBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(12)

        # Buttons you actually want
        self.btn_home = QPushButton("Home")
        self.btn_settings = QPushButton("Settings")

        self.buttons = [
            self.btn_home,
            self.btn_settings,
        ]

        for btn in self.buttons:
            btn.setMinimumHeight(45)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFlat(True)
            layout.addWidget(btn)

        self.setLayout(layout)
        self.setFixedWidth(0)
        self._collapsed = True

        self.anim = QPropertyAnimation(self, b"maximumWidth")
        self.anim.setDuration(300)

    def toggle(self):
        if self._collapsed:
            self.anim.setStartValue(0)
            self.anim.setEndValue(220)
        else:
            self.anim.setStartValue(220)
            self.anim.setEndValue(0)
        self.anim.start()
        self._collapsed = not self._collapsed

    def apply_theme(self, theme: str):
        if theme == "dark":
            style = """
                QPushButton {
                    background-color: #3a3a3a;
                    color: #f0f0f0;
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-size: 14px;
                    text-align: left;
                }
                QPushButton:hover { background-color: #505050; }
                QPushButton:pressed { background-color: #2d2d2d; }
            """
        else:
            style = """
                QPushButton {
                    background-color: #f5f5f5;
                    color: #222;
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-size: 14px;
                    text-align: left;
                }
                QPushButton:hover { background-color: #e0e0e0; }
                QPushButton:pressed { background-color: #d0d0d0; }
            """
        for btn in self.buttons:
            btn.setStyleSheet(style)
