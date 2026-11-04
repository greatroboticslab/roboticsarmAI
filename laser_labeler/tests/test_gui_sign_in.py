import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before PyQt6 is imported

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("PyQt6.QtWidgets")

import keyring  # noqa: E402
from keyring.backend import KeyringBackend  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from config import credentials  # noqa: E402
from config.settings import load_settings  # noqa: E402
from core.roboflow.auth import ProjectInfo, VerifyResult  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from gui.tabs.sign_in_tab import SignInTab  # noqa: E402

RF_KEY = "rf_FULLSECRETKEY_9876"
AI_KEY = "sk-FULLPROVIDERKEY-5555"


class Memory(KeyringBackend):
    priority = 1

    def __init__(self): self.d = {}
    def set_password(self, s, u, p): self.d[(s, u)] = p
    def get_password(self, s, u): return self.d.get((s, u))

    def delete_password(self, s, u):
        if (s, u) not in self.d:
            raise keyring.errors.PasswordDeleteError("missing")
        del self.d[(s, u)]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch):
    backend = Memory()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)
    for info in credentials.SERVICES.values():
        for name in info.env_vars:
            monkeypatch.delenv(name, raising=False)
    return backend


def make_tab(app, tmp_path, result=None, projects=None):
    result = result or VerifyResult(True, "valid", ["ws-one", "ws-two"], "ok")
    projects = projects if projects is not None else [ProjectInfo("proj-a", "Project A", "object-detection")]
    tab = SignInTab(user_settings_path=tmp_path / "user.yaml",
                    verify_fn=lambda key: result, projects_fn=lambda ws, key: projects)
    return tab


def wait_idle(app, tab, limit=5000):
    for _ in range(limit):
        app.processEvents()
        if not tab.is_busy():
            app.processEvents()
            return
        QApplication.instance().thread().msleep(1)
    raise AssertionError("background call did not finish")


def all_text(tab):
    return " | ".join(label.text() for label in tab.findChildren(QLabel))


def test_provider_save_stores_key_and_shows_only_mask(app, tmp_path, fake_keyring):
    tab = make_tab(app, tmp_path)
    seen = []
    tab.credentials_changed.connect(lambda: seen.append(1))
    row = tab.providers.rows["anthropic"]
    row.key_field.edit.setText(AI_KEY)
    row.save_btn.click()
    assert credentials.get_key("anthropic") == AI_KEY
    assert row.key_field.text() == ""  # field cleared after saving
    assert "5555" in row.status_label.text()
    assert AI_KEY not in all_text(tab) and AI_KEY[:-4] not in all_text(tab)
    assert seen == [1]


def test_provider_remove(app, tmp_path):
    tab = make_tab(app, tmp_path)
    row = tab.providers.rows["gemini"]
    row.key_field.edit.setText(AI_KEY)
    row.save_btn.click()
    row.remove_btn.click()
    assert credentials.get_key("gemini") is None
    assert "No key saved" in row.status_label.text()


def test_openai_compat_empty_key_is_not_an_error(app, tmp_path, fake_keyring):
    tab = make_tab(app, tmp_path)
    assert "Local servers (Ollama, LM Studio) usually need no key; leave empty." in all_text(tab)
    row = tab.providers.rows["openai_compat"]
    row.save_btn.click()
    assert "Nothing to save" in row.message_label.text()
    assert fake_keyring.d == {}


def test_roboflow_row_is_not_in_provider_list(app, tmp_path):
    assert set(make_tab(app, tmp_path).providers.rows) == {"openai_compat", "gemini", "anthropic"}


def test_key_fields_hide_text_until_show_is_ticked(app, tmp_path):
    from PyQt6.QtWidgets import QLineEdit
    tab = make_tab(app, tmp_path)
    kf = tab.roboflow.key_field
    assert kf.edit.echoMode() == QLineEdit.EchoMode.Password
    kf.show_box.setChecked(True)
    assert kf.edit.echoMode() == QLineEdit.EchoMode.Normal


def test_valid_roboflow_key_saved_and_choices_filled(app, tmp_path):
    tab = make_tab(app, tmp_path)
    box = tab.roboflow
    box.key_field.edit.setText(RF_KEY)
    box.verify_btn.click()
    assert not box.verify_btn.isEnabled()  # disabled while the call runs
    wait_idle(app, tab)
    assert box.verify_btn.isEnabled()
    assert credentials.get_key("roboflow") == RF_KEY
    assert box.key_field.text() == ""
    assert RF_KEY not in all_text(tab) and "9876" in box.status_label.text()
    assert [box.choice.workspace.itemText(i) for i in range(box.choice.workspace.count())] == ["ws-one", "ws-two"]
    assert box.choice.project.count() == 1
    box.choice.project.setCurrentIndex(0)
    box.choice.version.setText("3")
    box.choice.save_btn.click()
    rf = load_settings(tmp_path / "user.yaml")["roboflow"]
    assert rf == {"workspace": "ws-one", "project": "proj-a", "version": 3}


def test_invalid_roboflow_key_is_not_saved(app, tmp_path, fake_keyring):
    tab = make_tab(app, tmp_path, VerifyResult(False, "invalid", [], "Roboflow rejected this key (HTTP 401)."))
    tab.roboflow.key_field.edit.setText(RF_KEY)
    tab.roboflow.verify_btn.click()
    wait_idle(app, tab)
    assert credentials.get_key("roboflow") is None and fake_keyring.d == {}
    assert "not saved" in tab.roboflow.message_label.text()
    assert tab.roboflow.anyway_btn.isHidden()


def test_unreachable_offers_save_anyway(app, tmp_path):
    msg = "The key could NOT be checked: no connection to Roboflow."
    tab = make_tab(app, tmp_path, VerifyResult(False, "unreachable", [], msg))
    box = tab.roboflow
    box.key_field.edit.setText(RF_KEY)
    box.verify_btn.click()
    wait_idle(app, tab)
    assert "could not be checked" in box.message_label.text().lower()
    assert credentials.get_key("roboflow") is None
    assert not box.anyway_btn.isHidden()
    box.anyway_btn.click()
    assert credentials.get_key("roboflow") == RF_KEY
    assert box.key_field.text() == "" and box.anyway_btn.isHidden()
    assert RF_KEY not in all_text(tab)


def test_worker_exception_is_reported(app, tmp_path):
    def boom(key): raise RuntimeError("boom")
    tab = SignInTab(user_settings_path=tmp_path / "u.yaml", verify_fn=boom)
    tab.roboflow.key_field.edit.setText(RF_KEY)
    tab.roboflow.verify_btn.click()
    wait_idle(app, tab)
    assert "could not be checked" in tab.roboflow.message_label.text()
    assert tab.roboflow.verify_btn.isEnabled() and RF_KEY not in all_text(tab)


def test_key_with_spaces_rejected_in_plain_words(app, tmp_path):
    tab = make_tab(app, tmp_path)
    row = tab.providers.rows["gemini"]
    row.key_field.edit.setText("abc def")
    row.save_btn.click()
    assert "cannot be used" in row.message_label.text()


def test_no_keyring_explains_environment_variable(app, tmp_path, monkeypatch):
    from keyring.backends.fail import Keyring as Fail
    monkeypatch.setattr(keyring, "get_keyring", lambda: Fail())
    tab = make_tab(app, tmp_path)
    row = tab.providers.rows["anthropic"]
    row.key_field.edit.setText(AI_KEY)
    row.save_btn.click()
    text = row.message_label.text()
    assert "No system keyring was found" in text and "ANTHROPIC_API_KEY" in text
    assert AI_KEY not in all_text(tab)


def test_remove_roboflow_key_and_status_bar(app, tmp_path):
    tab = make_tab(app, tmp_path)  # fake verify/projects functions: no network
    win = MainWindow([("Sign in", lambda: tab)], tab.user_settings_path)
    tab.credentials_changed.connect(win.refresh_status)
    assert win.statusBar().currentMessage() == "Roboflow: not signed in"
    tab.roboflow.key_field.edit.setText(RF_KEY)
    tab.roboflow.verify_btn.click()
    wait_idle(app, tab)
    assert win.statusBar().currentMessage() == "Roboflow: signed in, no project chosen"
    tab.roboflow.choice.project.setCurrentIndex(0)
    tab.roboflow.choice.save_btn.click()
    assert win.statusBar().currentMessage() == "Roboflow: ws-one/proj-a"
    tab.roboflow.remove_btn.click()
    assert win.statusBar().currentMessage() == "Roboflow: not signed in"


def test_main_window_takes_tab_factories(app, tmp_path):
    from PyQt6.QtWidgets import QLabel as L
    win = MainWindow([("A", lambda: L("a")), ("B", lambda: L("b"))], tmp_path / "u.yaml")
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["A", "B"]
