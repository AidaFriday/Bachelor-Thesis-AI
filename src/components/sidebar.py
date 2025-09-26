from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QPropertyAnimation

class SideBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Layout (vertical, aligned top)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Buttons
        self.btn_home = QPushButton("Home")
        self.btn_dataset = QPushButton("Dataset")
        self.btn_models = QPushButton("Models")
        self.btn_benchmark = QPushButton("Benchmark")
        self.btn_settings = QPushButton("Settings")

        self.buttons = [
            self.btn_home,
            self.btn_dataset,
            self.btn_models,
            self.btn_benchmark,
            self.btn_settings,
        ]

        for btn in self.buttons:
            btn.setMinimumHeight(40)
            layout.addWidget(btn)

        self.setLayout(layout)
        self.setFixedWidth(0)  # start hidden
        self._collapsed = True

        # Animation for sliding
        self.anim = QPropertyAnimation(self, b"maximumWidth")
        self.anim.setDuration(300)  # ms

    def toggle(self):
        """Show/hide sidebar with slide animation"""
        if self._collapsed:
            self.anim.setStartValue(0)
            self.anim.setEndValue(200)
        else:
            self.anim.setStartValue(200)
            self.anim.setEndValue(0)
        self.anim.start()
        self._collapsed = not self._collapsed

    def apply_theme(self, theme: str):
        """Update button styles when theme changes."""
        if theme == "dark":
            style = """
                QPushButton {
                    background-color: #444;
                    color: #fff;
                    border: 1px solid #888;
                }
                QPushButton:hover {
                    background-color: #555;
                }
            """
        else:
            style = """
                QPushButton {
                    background-color: #f0f0f0;
                    color: #000;
                    border: 1px solid #888;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """
        for btn in self.buttons:
            btn.setStyleSheet(style)
