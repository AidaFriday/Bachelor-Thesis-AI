# components/sidebar.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFrame, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PyQt5.QtGui import QIcon


class NavButton(QPushButton):
    """
    A modern nav button with:
      - optional icon
      - "active" visual state (setActive)
    """
    def __init__(self, text: str, icon: QIcon = None, parent=None):
        super().__init__(text, parent)
        self._active = False

        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(False)
        self.setFlat(True)
        self.setMinimumHeight(44)

        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(self.iconSize() * 1.2)

        # Make text align left nicely
        self.setLayoutDirection(Qt.LeftToRight)

        # Keep the "real" label so we can hide/show cleanly
        self._full_text = text

    def set_full_text(self, text: str):
        self._full_text = text
        self.setText(text)

    def setActive(self, active: bool):
        self._active = active
        self.setProperty("active", active)
        # Force QSS refresh
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class SideBar(QWidget):
    """
    Modern elegant sidebar:
      - Card container (rounded, subtle border)
      - Header + nav sections
      - Active item highlight
      - Smooth easing animation
      - Clean theme application
    """
    COLLAPSED_W = 72    # icon-only vibe
    EXPANDED_W = 260

    def __init__(self, parent=None):
        super().__init__(parent)

        self._collapsed = True

        # Outer layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        # Card container
        self.card = QFrame()
        self.card.setObjectName("SideBarCard")
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(12)

        # Header (logo + title)
        header = QWidget()
        h = QHBoxLayout(header)
        h.setContentsMargins(8, 6, 8, 6)
        h.setSpacing(10)

        self.logo = QLabel("●")
        self.logo.setObjectName("SideBarLogo")

        self.title = QLabel("Dashboard")
        self.title.setObjectName("SideBarTitle")
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self._full_title = "Dashboard"

        h.addWidget(self.logo, 0, Qt.AlignVCenter)
        h.addWidget(self.title, 1)
        card_layout.addWidget(header)

        # Divider-like spacer line (subtle)
        divider = QFrame()
        divider.setObjectName("SideBarDivider")
        divider.setFixedHeight(1)
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        card_layout.addWidget(divider)

        # Buttons
        self.btn_home = NavButton("Home", icon=QIcon.fromTheme("go-home"))
        self.btn_settings = NavButton("Settings", icon=QIcon.fromTheme("settings"))

        self.buttons = [self.btn_home, self.btn_settings]
        for btn in self.buttons:
            btn.setObjectName("NavButton")
            card_layout.addWidget(btn)

        card_layout.addStretch(1)
        outer.addWidget(self.card, 1)

        # Start collapsed (icon-only look)
        self.setMinimumWidth(self.COLLAPSED_W)
        self.setMaximumWidth(self.COLLAPSED_W)

        # --- Animation: animate BOTH min/max in parallel ---
        self.anim_group = QParallelAnimationGroup(self)

        self.anim_max = QPropertyAnimation(self, b"maximumWidth")
        self.anim_min = QPropertyAnimation(self, b"minimumWidth")

        for a in (self.anim_max, self.anim_min):
            a.setDuration(280)
            a.setEasingCurve(QEasingCurve.InOutCubic)

        self.anim_group.addAnimation(self.anim_max)
        self.anim_group.addAnimation(self.anim_min)

        # When animation ends, finalize text visibility
        self.anim_group.finished.connect(self._on_anim_finished)

        # Default theme + active button
        self.apply_theme("dark")
        self.set_active(self.btn_home)

        # Ensure initial collapsed UI matches current state
        self._sync_collapsed_ui(collapsed=True)

    def toggle(self):
        expanding = self._collapsed  # if collapsed, we are expanding now

        start = self.maximumWidth()
        end = self.EXPANDED_W if expanding else self.COLLAPSED_W

        # During animation, don't fight layouts: lock both min/max to start first
        self.setMinimumWidth(start)
        self.setMaximumWidth(start)

        # If expanding, show text immediately so it doesn't "flash at end"
        # (it will be clipped naturally while width grows)
        if expanding:
            self._sync_collapsed_ui(collapsed=False)

        self.anim_group.stop()
        self.anim_max.setStartValue(start)
        self.anim_max.setEndValue(end)
        self.anim_min.setStartValue(start)
        self.anim_min.setEndValue(end)

        # Remember what state we are going to
        self._target_collapsed = not expanding

        self.anim_group.start()

        # Update state now (target state)
        self._collapsed = not expanding

    def _on_anim_finished(self):
        # After animation, enforce correct text visibility for target state
        self._sync_collapsed_ui(collapsed=getattr(self, "_target_collapsed", self._collapsed))

    def _sync_collapsed_ui(self, collapsed: bool):
        """
        collapsed=True => icon-only rail (no labels)
        collapsed=False => full labels
        """
        if collapsed:
            self.title.setText("")
            for b in self.buttons:
                b.setText("")
        else:
            self.title.setText(self._full_title)
            # restore each button text
            self.btn_home.setText(self.btn_home._full_text)
            self.btn_settings.setText(self.btn_settings._full_text)

    def set_active(self, button: QPushButton):
        for b in self.buttons:
            if hasattr(b, "setActive"):
                b.setActive(b is button)

    def apply_theme(self, theme: str):
        """
        Modern QSS:
          - Card with rounded corners, subtle border
          - Buttons with hover glow + active pill indicator
        """
        if theme == "dark":
            qss = """
            /* ---- Card ---- */
            QFrame#SideBarCard {
                background: rgba(30, 30, 34, 0.92);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
            }

            QLabel#SideBarLogo {
                color: #9ef0c8;
                font-size: 18px;
                padding-left: 2px;
            }
            QLabel#SideBarTitle {
                color: #f3f4f6;
                font-size: 15px;
                font-weight: 600;
                letter-spacing: 0.2px;
            }

            QFrame#SideBarDivider {
                background: rgba(255,255,255,0.08);
                border: none;
            }

            /* ---- Nav buttons ---- */
            QPushButton#NavButton {
                color: rgba(243, 244, 246, 0.92);
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 14px;
                padding: 10px 12px;
                text-align: left;
                font-size: 13px;
                font-weight: 500;
                margin-top: 2px;
            }
            QPushButton#NavButton:hover {
                background: rgba(158, 240, 200, 0.10);
                border: 1px solid rgba(158, 240, 200, 0.18);
            }
            QPushButton#NavButton:pressed {
                background: rgba(158, 240, 200, 0.14);
            }

            /* Active state (property-based) */
            QPushButton#NavButton[active="true"] {
                background: rgba(158, 240, 200, 0.16);
                border: 1px solid rgba(158, 240, 200, 0.28);
            }
            """
        else:
            qss = """
            QFrame#SideBarCard {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(0, 0, 0, 0.08);
                border-radius: 18px;
            }

            QLabel#SideBarLogo {
                color: #0aa37f;
                font-size: 18px;
                padding-left: 2px;
            }
            QLabel#SideBarTitle {
                color: #111827;
                font-size: 15px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }

            QFrame#SideBarDivider {
                background: rgba(0,0,0,0.08);
                border: none;
            }

            QPushButton#NavButton {
                color: rgba(17, 24, 39, 0.92);
                background: rgba(0, 0, 0, 0.03);
                border: 1px solid rgba(0, 0, 0, 0.06);
                border-radius: 14px;
                padding: 10px 12px;
                text-align: left;
                font-size: 13px;
                font-weight: 600;
                margin-top: 2px;
            }
            QPushButton#NavButton:hover {
                background: rgba(10, 163, 127, 0.10);
                border: 1px solid rgba(10, 163, 127, 0.18);
            }
            QPushButton#NavButton:pressed {
                background: rgba(10, 163, 127, 0.14);
            }

            QPushButton#NavButton[active="true"] {
                background: rgba(10, 163, 127, 0.16);
                border: 1px solid rgba(10, 163, 127, 0.28);
            }
            """

        self.setStyleSheet(qss)
