"""Shared pieces for the GUI tests: an offscreen app, a throwaway project folder and a context on it."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402

from gui.paths import PROJECT_ROOT  # noqa: E402
from tests.synth import finish, rect, scene, spot  # noqa: E402


def make_project(tmp_path: Path) -> Path:
    """A copy of the taxonomy and prompt parts, so tests never touch the real project files."""
    root = tmp_path / "project"
    shutil.copytree(PROJECT_ROOT / "taxonomy", root / "taxonomy")
    shutil.copytree(PROJECT_ROOT / "prompts" / "parts", root / "prompts" / "parts")
    (root / "prompts" / "versions").mkdir(parents=True)
    return root


def make_images(folder: Path, names=("a.png", "b.png", "c.png")) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        img = scene()
        rect(img, 100, 80, 80, 60, (150, 30, 30))
        spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
        Image.fromarray(finish(img)).save(folder / name)
        paths.append(folder / name)
    return paths


def wait_until(app, condition, timeout=10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(condition())
