import os
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QHBoxLayout,
)


class SelectSubjectsDialog(QDialog):
    def __init__(self, dataset_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select People")
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

        layout.addWidget(self.list_widget)

        buttons = QHBoxLayout()
        select_all_btn = QPushButton("All")
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")

        select_all_btn.clicked.connect(self._select_all)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        buttons.addWidget(select_all_btn)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

    def accept(self):
        self.selected_subjects = [
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState()
        ]
        super().accept()

    def _select_all(self):
        self.selected_subjects = ["__ALL__"]
        super().accept()
