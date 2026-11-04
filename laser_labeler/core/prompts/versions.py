"""Numbered, immutable prompt versions (spec section 7).

prompts/versions/v0001/ holds a copy of every part plus version.json. The editable
draft stays in prompts/parts/. `ACTIVE` names the version used for runs. A saved
version is never changed; to change the prompt, save a new one.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

VERSION_RE = re.compile(r"^v\d{4}$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VersionInfo:
    version: str
    created: str
    note: str
    parent: str | None
    path: Path


def _read_parts(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.md"))}


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def get_version_dir(versions_dir: str | Path, version: str) -> Path:
    """Folder of a saved version. The id must look like v0001; nothing else is turned into a path."""
    if not isinstance(version, str) or not VERSION_RE.match(version):
        raise ValueError(f"{version!r} is not a prompt version id (expected something like v0001)")
    path = Path(versions_dir) / version
    if not path.is_dir():
        raise ValueError(f"prompt version {version} does not exist")
    return path


def list_versions(versions_dir: str | Path) -> list[VersionInfo]:
    """Saved versions, oldest first. Folders without a valid version.json are ignored."""
    found = []
    for path in sorted(Path(versions_dir).glob("v*")) if Path(versions_dir).is_dir() else []:
        if not (VERSION_RE.match(path.name) and path.is_dir()):
            continue
        try:
            meta = json.loads((path / "version.json").read_text(encoding="utf-8"))
            found.append(VersionInfo(meta["version"], meta["created"], meta.get("note", ""),
                                     meta.get("parent"), path))
        except (OSError, ValueError, KeyError):
            continue
    return found


def active_version(versions_dir: str | Path) -> str | None:
    try:
        version = (Path(versions_dir) / "ACTIVE").read_text(encoding="utf-8").strip()
        get_version_dir(versions_dir, version)
    except (OSError, ValueError):
        return None
    return version


def set_active(versions_dir: str | Path, version: str) -> None:
    get_version_dir(versions_dir, version)
    (Path(versions_dir) / "ACTIVE").write_text(version + "\n", encoding="utf-8")


def save_version(parts_dir: str | Path, versions_dir: str | Path, note: str = "",
                 parent: str | None = None, taxonomy: dict | None = None) -> str:
    """Save the current draft as the next version. Identical to the newest version: reuse it.

    `taxonomy` (optional, as a plain dict) is stored beside the parts so "Generate new prompt" can later tell
    what was added since this version was written."""
    parts = _read_parts(Path(parts_dir))
    if not parts:
        raise ValueError(f"no prompt parts found in {parts_dir}")
    versions_dir = Path(versions_dir)
    existing = list_versions(versions_dir)
    if existing and _read_parts(existing[-1].path) == parts:
        return existing[-1].version
    numbers = [int(p.name[1:]) for p in versions_dir.glob("v*") if VERSION_RE.match(p.name)] \
        if versions_dir.is_dir() else []
    version = f"v{max(numbers, default=0) + 1:04d}"
    target = versions_dir / version
    target.mkdir(parents=True)
    for name, data in parts.items():
        (target / name).write_bytes(data)
    meta = {"version": version, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": note, "parent": parent if parent is not None else active_version(versions_dir),
            "parts": {name: _sha(data) for name, data in parts.items()}}
    (target / "version.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if taxonomy is not None:
        (target / "taxonomy.json").write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    return version


def taxonomy_snapshot(versions_dir: str | Path, version: str) -> dict | None:
    """The taxonomy as it was when this version was saved, or None if none was stored."""
    try:
        return json.loads((get_version_dir(versions_dir, version) / "taxonomy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def resolve_parts_dir(parts_dir: str | Path, versions_dir: str | Path) -> Path:
    """The active version's folder if there is one, otherwise the editable draft."""
    active = active_version(versions_dir)
    return get_version_dir(versions_dir, active) if active else Path(parts_dir)


def draft_differs_from_active(parts_dir: str | Path, versions_dir: str | Path) -> bool:
    """True when the editable parts have changes that are not saved in the active version."""
    draft = _read_parts(Path(parts_dir))
    active = active_version(versions_dir)
    if active is None:
        return bool(draft)
    return draft != _read_parts(get_version_dir(versions_dir, active))


def diff_versions(versions_dir: str | Path, a: str, b: str, parts_dir: str | Path | None = None) -> str:
    """Unified diff of every part between two versions. `b` (or `a`) may be "draft"."""
    def load(which: str) -> dict[str, bytes]:
        if which == "draft":
            if parts_dir is None:
                raise ValueError("parts_dir is needed to diff against the draft")
            return _read_parts(Path(parts_dir))
        return _read_parts(get_version_dir(versions_dir, which))

    left, right = load(a), load(b)
    out: list[str] = []
    for name in sorted(set(left) | set(right)):
        old = left.get(name, b"").decode("utf-8").splitlines(keepends=True)
        new = right.get(name, b"").decode("utf-8").splitlines(keepends=True)
        out.extend(difflib.unified_diff(old, new, f"{a}/{name}" if name in left else "/dev/null",
                                        f"{b}/{name}" if name in right else "/dev/null"))
    return "".join(out)


def prompt_dir_for_run(project_root: str | Path = PROJECT_ROOT) -> Path:
    root = Path(project_root)
    return resolve_parts_dir(root / "prompts" / "parts", root / "prompts" / "versions")
