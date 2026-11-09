"""AI settings: provider, model, run mode, spend cap, hybrid escalation and the Test provider button (spec section 10)."""
from __future__ import annotations

from copy import deepcopy

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from config import credentials
from config.credentials import CredentialError, MissingCredentialError
from config.settings import SettingsError
from core.labeler.factory import ESCALATION_PROVIDERS
from core.providers.check import check_provider, format_check
from core.providers.hints import clean_model_name
from gui.context import AppContext
from gui.widgets.messages import describe_error
from gui.workers.verify_worker import start_call

PROVIDER_LABELS = {"openai_compat": "OpenAI-compatible (local server, OpenAI, OpenRouter)", "gemini": "Google Gemini",
                   "anthropic": "Claude (Anthropic)"}
SCHEME_LABELS = {"full": "Full: obj_... classes and las_<event> classes (default)",
                 "components": "Components: obj_... classes, and laser spots as component1 (direct hit) / component2 (bounce)",
                 "merged": "Merged: no separate object boxes; each laser spot is named after its object, "
                           "e.g. obj_pen_rubber_red_first__component1"}
MATCH_LABELS = {"off": "Off: reuse a listed object or material if one fits, otherwise add the new one to your lists (good while filling them up)",
                "object": "Object only: the object name must be on your list; material and color are taken as the model says",
                "all": "Everything: object, material, color and incidence must all be on your lists"}
RULE_LABELS = {"only_if_no_object_hit": "Label the table only if no object has a laser hit (default)",
               "always_label": "Always label the table", "never_label": "Never label the table"}


def _combo(options: dict[str, str]) -> QComboBox:
    box = QComboBox()
    for value, label in options.items():
        box.addItem(label, value)
    return box


def _spin(low: float, high: float, decimals: int, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    return spin


class AISettingsTab(QWidget):
    go_sign_in = pyqtSignal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._runner = None
        self._profiles: dict[str, dict] = {}  # model name and prices remembered for each provider
        self._shown: str | None = None  # the provider whose values the model and price boxes currently show
        self._loading = False
        self.provider = _combo(PROVIDER_LABELS)
        self.model = QLineEdit()
        self.model.setPlaceholderText("Copy the exact model name from your provider (nothing is built in)")
        self.base_url = QLineEdit()
        self.run_mode = _combo({"pipeline": "Pipeline: the app finds spots, the model classifies (best for local models)",
                                "agent": "Agent: the model calls the tools itself (needs a tool-capable model)"})
        self.structured = QCheckBox("Ask the server to force valid JSON (Ollama, LM Studio, OpenAI; pipeline mode only)")
        self.temperature = _spin(0, 2, 2, 0.1)
        self.disable_thinking = QCheckBox("Turn off thinking (qwen3 and other reasoning models on Ollama; much faster)")
        self.timeout = QSpinBox()
        self.timeout.setRange(0, 3600)
        self.timeout.setSingleStep(30)
        self.timeout.setSpecialValueText("server default")
        self.timeout.setSuffix(" s")
        self.timeout.setToolTip("Give up on one slow request after this long and mark the image failed (so a slow model "
                                "cannot stall the queue). Not retried.")
        self.max_reply = QSpinBox()
        self.max_reply.setRange(0, 32768)
        self.max_reply.setSingleStep(100)
        self.max_reply.setSpecialValueText("no limit")
        self.max_reply.setSuffix(" tokens")
        self.max_reply.setToolTip("Stop a reply after this many tokens. The answer the app needs is short (a few hundred "
                                  "tokens), so 1200 makes a model that rambles or thinks at length fail fast instead of "
                                  "stalling. If your server answers with an error about max_tokens, set it back to no limit.")
        self.auto_resume = QCheckBox("When the provider's limit is reached, wait and carry on by itself")
        self.auto_resume.setToolTip("Waits as long as the provider says (or tries again after 1, 2, 5, 10, 20, then every 30 "
                                    "minutes) and keeps going, so a queue can run overnight when a daily quota reopens. "
                                    "Turn off to be asked each time instead.")
        self.auto_fit = QCheckBox("Fit the request to the context window (retry smaller if the server says it is too long)")
        self.max_spots = QSpinBox()
        self.max_spots.setRange(1, 12)
        self.max_spots.setToolTip("Each candidate laser spot adds a zoomed crop to the request. Fewer spots is faster.")
        self.crop_px = QSpinBox()
        self.crop_px.setRange(64, 1024)
        self.crop_px.setSingleStep(32)
        self.crop_px.setToolTip("Longest side of each zoomed crop. Smaller is faster and uses less context.")
        self.max_px = QSpinBox()
        self.max_px.setRange(256, 4096)
        self.max_px.setSingleStep(128)
        self.price_in, self.price_out = _spin(0, 10000, 4, 0.1), _spin(0, 10000, 4, 0.1)
        self.cap = _spin(0, 100000, 2, 1.0)
        self.key_status = QLabel()
        self.key_status.setWordWrap(True)
        self.sign_in_btn = QPushButton("Enter or change keys on the Sign in tab")
        self.table_rule = _combo(RULE_LABELS)
        self.match = _combo(MATCH_LABELS)
        self.scheme = _combo(SCHEME_LABELS)
        self.scheme_help = QLabel("component1 = a direct hit: the beam lands on the object (two direct hits on the same object are both "
                                  "component1). component2 = every bounce: light reflected, refracted or scattered "
                                  "off something. Every laser spot gets one of the two; there are no halo or beam boxes. "
                                  "Applies to images labeled from now on.")
        self.scheme_help.setWordWrap(True)
        self.box_size = _spin(0, 0.5, 2, 0.05)
        self.box_size.setToolTip("0 keeps the laser box tight around the spot. 0.33 makes it a third of the object box it hits.")
        self.keep_awake = QCheckBox("Keep the computer awake during a labeling run")
        self.esc_on = QCheckBox("Send images the main model fails on or is unsure about to a cloud model")
        self.esc_provider = _combo({k: PROVIDER_LABELS[k] for k in ESCALATION_PROVIDERS})
        self.esc_model = QLineEdit()
        self.esc_threshold = _spin(0, 1, 2, 0.05)
        self.esc_in, self.esc_out = _spin(0, 10000, 4, 0.1), _spin(0, 10000, 4, 0.1)
        self.esc_key_status = QLabel()
        self.save_btn, self.test_btn = QPushButton("Save settings"), QPushButton("Test provider")
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(130)
        self.report.setPlaceholderText("The test result appears here.")
        self._build()
        self.provider.currentIndexChanged.connect(self._provider_changed)
        self.esc_on.toggled.connect(self._refresh_visibility)
        self.esc_provider.currentIndexChanged.connect(self._refresh_visibility)
        self.sign_in_btn.clicked.connect(self.go_sign_in)
        self.save_btn.clicked.connect(self.save)
        self.test_btn.clicked.connect(self.test_provider)
        ctx.settings_changed.connect(self.load)
        self.load()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        main = QGroupBox("Main model")
        form = QFormLayout(main)
        form.addRow("Provider", self.provider)
        form.addRow("Model", self.model)
        self.base_url_label = QLabel("Server address")
        form.addRow(self.base_url_label, self.base_url)
        form.addRow("Run mode", self.run_mode)
        form.addRow("", self.structured)
        form.addRow("", self.disable_thinking)
        form.addRow("Temperature", self.temperature)
        form.addRow("Largest image side sent (px)", self.max_px)
        form.addRow("", self.auto_fit)
        form.addRow("Spots shown to the model per image", self.max_spots)
        form.addRow("Zoom crop size (px)", self.crop_px)
        self.timeout_label = QLabel("Request timeout")
        form.addRow(self.timeout_label, self.timeout)
        self.max_reply_label = QLabel("Longest reply allowed")
        form.addRow(self.max_reply_label, self.max_reply)
        form.addRow("", self.auto_resume)
        form.addRow("Key", self.key_status)
        form.addRow("", self.sign_in_btn)
        money = QGroupBox("Spend (you enter your provider's prices; 0 means free or local)")
        mform = QFormLayout(money)
        mform.addRow("USD per million input tokens", self.price_in)
        mform.addRow("USD per million output tokens", self.price_out)
        mform.addRow("Daily spend cap (USD)", self.cap)
        rules = QGroupBox("Labeling rules")
        rform = QFormLayout(rules)
        rform.addRow("Taxonomy match", self.match)
        rform.addRow("Label scheme", self.scheme)
        rform.addRow("", self.scheme_help)
        rform.addRow("Laser box size (fraction of the object it hits, 0 = tight)", self.box_size)
        rform.addRow("Table rule", self.table_rule)
        rform.addRow("", self.keep_awake)
        self.esc_box = QGroupBox("Hybrid escalation (off by default)")
        eform = QFormLayout(self.esc_box)
        eform.addRow(self.esc_on)
        eform.addRow("Cloud provider", self.esc_provider)
        eform.addRow("Cloud model", self.esc_model)
        eform.addRow("Escalate below confidence", self.esc_threshold)
        eform.addRow("USD per million input tokens", self.esc_in)
        eform.addRow("USD per million output tokens", self.esc_out)
        eform.addRow("Key", self.esc_key_status)
        inner = QWidget()
        col = QVBoxLayout(inner)
        for box in (main, money, rules, self.esc_box):
            col.addWidget(box)
        col.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.test_btn)
        buttons.addStretch(1)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll, 1)
        outer.addLayout(buttons)
        outer.addWidget(self.message)
        outer.addWidget(self.report)

    # ------------------------------------------------------------------ settings <-> form
    @staticmethod
    def _pick(box: QComboBox, value: str) -> None:
        index = box.findData(value)
        box.setCurrentIndex(index if index >= 0 else 0)

    def load(self) -> None:
        try:
            s = self.ctx.settings()
        except SettingsError as exc:
            self.message.setText(f"Your settings file has a problem: {exc}")
            return
        p, e, lab = s["provider"], s["escalation"], s["labeling"]
        self._loading = True
        self._profiles = deepcopy(p.get("profiles") or {})
        # the active model and prices are the truth (the command line or a hand edit may have changed them)
        self._profiles[p["provider"]] = {"model": p.get("model", ""), "price_in_per_mtok": p.get("price_in_per_mtok", 0),
                                         "price_out_per_mtok": p.get("price_out_per_mtok", 0)}
        self._shown = p["provider"]
        self._pick(self.provider, p["provider"])
        self.model.setText(p.get("model", ""))
        self.base_url.setText(p.get("base_url", ""))
        self._pick(self.run_mode, p["run_mode"])
        self.structured.setChecked(bool(p.get("structured_output")))
        self.disable_thinking.setChecked(bool(p.get("disable_thinking")))
        self.timeout.setValue(int(p.get("timeout_s", 300)))
        self.max_reply.setValue(int(p.get("reply_limit", 0) or 0))
        self.auto_resume.setChecked(bool(s["run"].get("auto_resume_on_limit", True)))
        self.auto_fit.setChecked(bool(s["run"].get("auto_fit_context", True)))
        self.max_spots.setValue(int(s["run"].get("max_candidates", 6)))
        self.crop_px.setValue(int(s["run"].get("crop_px", 384)))
        self.temperature.setValue(float(p.get("temperature", 0.0)))
        self.max_px.setValue(int(p.get("max_image_px", 1024)))
        self.price_in.setValue(float(p.get("price_in_per_mtok", 0)))
        self.price_out.setValue(float(p.get("price_out_per_mtok", 0)))
        self.cap.setValue(float(p.get("daily_cost_cap_usd", 0)))
        self._pick(self.table_rule, lab["table_rule"])
        self._pick(self.match, lab.get("taxonomy_match", "object"))
        self._pick(self.scheme, lab.get("scheme", "full"))
        self.box_size.setValue(float(lab.get("laser_box_object_fraction", 0.0)))
        self.keep_awake.setChecked(bool(lab.get("keep_awake")))
        self.esc_on.setChecked(bool(e.get("enabled")))
        self._pick(self.esc_provider, e.get("provider") or ESCALATION_PROVIDERS[0])
        self.esc_model.setText(e.get("model", ""))
        self.esc_threshold.setValue(float(e.get("confidence_threshold", 0.5)))
        self.esc_in.setValue(float(e.get("price_in_per_mtok", 0)))
        self.esc_out.setValue(float(e.get("price_out_per_mtok", 0)))
        self._loading = False
        self._refresh_visibility()

    def _stash(self) -> None:
        """Remember what the model and price boxes hold under the provider they belong to."""
        if self._shown:
            self._profiles[self._shown] = {"model": clean_model_name(self.model.text()),
                                           "price_in_per_mtok": self.price_in.value(),
                                           "price_out_per_mtok": self.price_out.value()}

    def _provider_changed(self) -> None:
        """Switching provider shows the model name and prices last used with that provider (empty the first time)."""
        if not self._loading:
            self._stash()
            now = self.provider.currentData()
            saved = self._profiles.get(now, {})
            self.model.setText(saved.get("model", ""))
            self.price_in.setValue(float(saved.get("price_in_per_mtok", 0)))
            self.price_out.setValue(float(saved.get("price_out_per_mtok", 0)))
            self._shown = now
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        local = self.provider.currentData() == "openai_compat"
        for widget in (self.base_url, self.base_url_label, self.structured, self.disable_thinking, self.timeout,
                       self.timeout_label, self.max_reply, self.max_reply_label):
            widget.setVisible(local)
        for box, label in ((self.provider, self.key_status), (self.esc_provider, self.esc_key_status)):
            label.setText(self._key_text(box.currentData()))
        for widget in (self.esc_provider, self.esc_model, self.esc_threshold, self.esc_in, self.esc_out):
            widget.setEnabled(self.esc_on.isChecked())

    @staticmethod
    def _key_text(service: str) -> str:
        try:
            source, masked = credentials.key_source(service), credentials.mask_key(credentials.get_key(service))
        except CredentialError as exc:
            return str(exc)
        if source == "missing":
            return "No key saved" + (" (optional for local servers)" if service == "openai_compat" else ": add it on the Sign in tab")
        return f"Key found ({source}): {masked}"

    def provider_block(self) -> dict:
        """The provider settings as shown in the form (used for saving and for Test provider)."""
        return {"provider": self.provider.currentData(), "model": clean_model_name(self.model.text()),
                "base_url": self.base_url.text().strip(), "run_mode": self.run_mode.currentData(),
                "structured_output": self.structured.isChecked(), "temperature": self.temperature.value(),
                "disable_thinking": self.disable_thinking.isChecked(), "timeout_s": self.timeout.value(),
                "reply_limit": self.max_reply.value(),
                "max_image_px": self.max_px.value(), "price_in_per_mtok": self.price_in.value(),
                "price_out_per_mtok": self.price_out.value(), "daily_cost_cap_usd": self.cap.value()}

    def save(self) -> bool:
        self._stash()
        profiles = deepcopy(self._profiles)

        def change(s: dict) -> None:
            s["provider"].update(self.provider_block())
            s["provider"]["profiles"] = profiles
            s["run"]["auto_fit_context"] = self.auto_fit.isChecked()
            s["run"]["auto_resume_on_limit"] = self.auto_resume.isChecked()
            s["run"]["max_candidates"] = self.max_spots.value()
            s["run"]["crop_px"] = self.crop_px.value()
            s["labeling"]["table_rule"] = self.table_rule.currentData()
            s["labeling"]["taxonomy_match"] = self.match.currentData()
            s["labeling"]["scheme"] = self.scheme.currentData()
            s["labeling"]["laser_box_object_fraction"] = round(self.box_size.value(), 2)
            s["labeling"]["keep_awake"] = self.keep_awake.isChecked()
            s["escalation"].update(enabled=self.esc_on.isChecked(), provider=self.esc_provider.currentData(),
                                   model=self.esc_model.text().strip(), confidence_threshold=self.esc_threshold.value(),
                                   price_in_per_mtok=self.esc_in.value(), price_out_per_mtok=self.esc_out.value())
            if self.esc_on.isChecked() and not self.esc_model.text().strip():
                raise SettingsError("escalation is on but no cloud model name is set")
        try:
            self.ctx.update_settings(change)
        except (SettingsError, OSError) as exc:
            self.message.setText(describe_error(exc))
            return False
        self.message.setText("Saved. The next labeling run uses these settings.")
        return True

    # ------------------------------------------------------------------ test provider
    def test_provider(self) -> None:
        self.report.clear()
        try:
            provider = self.ctx.provider(self.provider_block())
        except (MissingCredentialError, ValueError, CredentialError) as exc:
            self.message.setText(f"Cannot test yet: {exc}")
            return
        self.test_btn.setEnabled(False)
        self.message.setText("Testing: a few small requests are being sent...")
        label = f"{provider.name}/{provider.model}"
        self._runner = start_call(check_provider, (provider,), lambda result: self._tested(label, result), self._test_failed, self)

    def _tested(self, label: str, result) -> None:
        self.test_btn.setEnabled(True)
        self.report.setPlainText(format_check(label, result))
        self.message.setText("Test finished." if result.valid_json_rate > 0 else
                             "The model never returned valid JSON. Check the model name and server address.")

    def _test_failed(self, message: str) -> None:
        self.test_btn.setEnabled(True)
        self.message.setText(f"The test could not run: {message}")
