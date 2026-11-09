"""Main window: the six tabs in spec order, one shared AppContext, and a status bar.

The status bar shows the Roboflow project (as its message) and, as permanent items, the prompt version,
the provider and model, and spend so far today. Tests can still pass their own (title, factory) tabs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QLabel, QMainWindow, QTabWidget, QWidget

from config import credentials
from config.settings import load_defaults, load_settings
from gui.context import AppContext
from gui.paths import default_user_settings_path
from gui.tabs.sign_in_tab import SignInTab

TabFactory = Callable[[], QWidget]
SIGN_IN, SELECT, RUN, REVIEW, PROMPTS, AI = "Sign in", "Select images", "Label run", "Review queue", "Prompts and taxonomy", "AI settings"


def roboflow_status_text(settings_path: Path) -> str:
    """'Roboflow: workspace/project', or 'not signed in'."""
    if not credentials.get_key("roboflow"):
        return "Roboflow: not signed in"
    try:
        rf = load_settings(settings_path)["roboflow"]
    except Exception:
        rf = load_defaults()["roboflow"]
    if rf.get("workspace") and rf.get("project"):
        return f"Roboflow: {rf['workspace']}/{rf['project']}"
    return "Roboflow: signed in, no project chosen"


def provider_status_text(settings: dict) -> str:
    block = settings["provider"]
    model = block.get("model") or "no model chosen"
    return f"AI: {block['provider']} / {model}"


def spend_status_text(spent: float, settings: dict) -> str:
    cap = settings["provider"].get("daily_cost_cap_usd")
    return f"Spent today: ${spent:.2f}" + (f" of ${cap:.2f}" if cap else "")


class MainWindow(QMainWindow):
    def __init__(self, tab_factories: list[tuple[str, TabFactory]] | None = None,
                 user_settings_path: Path | None = None, *, ctx: AppContext | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Laser Labeler")
        self.resize(1200, 820)
        self.settings_path = Path(user_settings_path or default_user_settings_path())
        self.ctx: AppContext | None = ctx
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.sign_in: SignInTab | None = None
        self.prompt_label = self.provider_label = self.spend_label = None
        if tab_factories is None:
            self.ctx = self.ctx or AppContext(self.settings_path)
            tab_factories = self._full_app_factories(self.ctx)
        for title, factory in tab_factories:
            self.tabs.addTab(factory(), title)
        if self.ctx is not None:
            self._build_status_items(self.ctx)
        self.refresh_status()

    # ------------------------------------------------------------------ tabs
    def _full_app_factories(self, ctx: AppContext) -> list[tuple[str, TabFactory]]:
        from gui.tabs.ai_settings_tab import AISettingsTab
        from gui.tabs.label_run_tab import LabelRunTab
        from gui.tabs.prompts_tab import PromptsTab
        from gui.tabs.review_tab import ReviewTab
        from gui.tabs.select_images_tab import SelectImagesTab

        def ai() -> AISettingsTab:
            tab = AISettingsTab(ctx)
            tab.go_sign_in.connect(lambda: self.show_tab(SIGN_IN))
            return tab

        def review() -> ReviewTab:
            tab = ReviewTab(ctx)
            ctx.open_image.connect(lambda image_id: (self.show_tab(REVIEW), tab.select_image(image_id)))
            return tab

        return [(SIGN_IN, self._make_sign_in), (SELECT, lambda: SelectImagesTab(ctx)), (RUN, lambda: LabelRunTab(ctx)),
                (REVIEW, review), (PROMPTS, lambda: PromptsTab(ctx)), (AI, ai)]

    def _make_sign_in(self) -> SignInTab:
        self.sign_in = SignInTab(user_settings_path=self.settings_path)
        self.sign_in.credentials_changed.connect(self._credentials_changed)
        return self.sign_in

    def show_tab(self, title: str) -> None:
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == title:
                self.tabs.setCurrentIndex(i)
                return

    # ------------------------------------------------------------------ status bar
    def _build_status_items(self, ctx: AppContext) -> None:
        self.prompt_label, self.provider_label, self.spend_label = QLabel(), QLabel(), QLabel()
        for label in (self.prompt_label, self.provider_label, self.spend_label):
            self.statusBar().addPermanentWidget(label)
        ctx.settings_changed.connect(self.refresh_status)
        ctx.prompt_changed.connect(self.refresh_status)
        ctx.queue_changed.connect(self.refresh_status)
        self._timer = QTimer(self)  # spend changes while a run is going
        self._timer.setInterval(5000)
        self._timer.timeout.connect(self.refresh_status)
        self._timer.start()

    def _credentials_changed(self) -> None:
        if self.ctx is not None:
            self.ctx.settings_changed.emit()  # the provider choice may now be usable
        self.refresh_status()

    def refresh_status(self) -> None:
        self.statusBar().showMessage(roboflow_status_text(self.settings_path))
        if self.ctx is None:
            return
        try:
            settings = self.ctx.settings()
            self.prompt_label.setText(f"Prompt: {self.ctx.prompt_version_text()}")
            self.provider_label.setText(provider_status_text(settings))
            self.spend_label.setText(spend_status_text(self.ctx.tracker().spent_today, settings))
        except Exception as exc:  # the status bar must never break the window
            self.spend_label.setText(f"status unavailable: {exc}")

    def closeEvent(self, event: QCloseEvent) -> None:
        for i in range(self.tabs.count()):
            stop = getattr(self.tabs.widget(i), "shutdown", None)
            if callable(stop):
                stop()  # a labeling run (or its wait for a provider limit) must end before the queue closes
        if self.ctx is not None:
            self.ctx.close()
        super().closeEvent(event)
