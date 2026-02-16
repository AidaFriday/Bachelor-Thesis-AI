from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout


class SelectMetricDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Evaluation Method")

        layout = QVBoxLayout(self)

        label = QLabel(
            "Which metric would you like to compute?\n"
            "• ROC Curve\n"
            "• Confusion Matrix"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        buttons = QHBoxLayout()
        btn_roc = QPushButton("ROC Curve")
        btn_cm = QPushButton("Confusion Matrix")

        btn_roc.clicked.connect(lambda: self._choose("roc"))
        btn_cm.clicked.connect(lambda: self._choose("cm"))

        buttons.addWidget(btn_roc)
        buttons.addWidget(btn_cm)
        layout.addLayout(buttons)

        self.selection = None

    def _choose(self, selection):
        self.selection = selection
        self.accept()
