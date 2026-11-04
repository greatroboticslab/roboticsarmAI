"""Corrections the user makes in review, kept as examples that can be added to the prompt (spec section 7).

Each correction is one line of data/corrections.jsonl. Nothing goes into the prompt until the user picks it:
`apply_to_background` writes the chosen ones into a marked block of the background part, replacing the
previous block, so repeated use never piles up duplicates.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from core.review.diffs import Change

BACKGROUND_PART = "07_background.md"
START, END = "<!-- corrections:start -->", "<!-- corrections:end -->"
USEFUL_KINDS = ("attribute", "added", "removed", "box")  # hand-edited numbers are not teachable examples


def correction_id(image_id: str, text: str) -> str:
    return hashlib.sha1(f"{image_id}|{text}".encode("utf-8")).hexdigest()[:10]


def load_corrections(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue  # a damaged line must not lose the rest
        if isinstance(item, dict) and "id" in item:
            items.append(item)
    return items


def _write(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    tmp.replace(path)


def record_corrections(path: str | Path, image_id: str, image_name: str, changes: Sequence[Change],
                       prompt_version: str = "") -> int:
    """Add the useful changes of one edit. The same correction on the same image is stored once. Returns how many are new."""
    path = Path(path)
    items = load_corrections(path)
    known = {i["id"] for i in items}
    added = 0
    for change in changes:
        if change.kind not in USEFUL_KINDS:
            continue
        text = change.describe()
        cid = correction_id(image_id, text)
        if cid in known:
            continue
        known.add(cid)
        items.append({"id": cid, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "image_id": image_id, "image_name": image_name, "prompt_version": prompt_version,
                      "text": text, "change": change.to_dict(), "included": False})
        added += 1
    if added:
        _write(path, items)
    return added


def set_included(path: str | Path, ids: Iterable[str], included: bool = True) -> int:
    path, wanted = Path(path), set(ids)
    items = load_corrections(path)
    count = 0
    for item in items:
        if item["id"] in wanted and item.get("included") != included:
            item["included"] = included
            count += 1
    if count:
        _write(path, items)
    return count


def render_block(items: Sequence[dict]) -> str:
    lines = [START, "Past corrections from the reviewer (the model's first answer was wrong; the reviewer's fix is after the arrow):"]
    lines += [f"- {item['text']}" for item in items]
    lines.append(END)
    return "\n".join(lines)


def apply_to_background(parts_dir: str | Path, items: Sequence[dict]) -> bool:
    """Put the given corrections in the marked block of the background part (empty list removes the block).
    Returns True if the file changed."""
    part = Path(parts_dir) / BACKGROUND_PART
    text = part.read_text(encoding="utf-8") if part.is_file() else ""
    if not items and not (START in text and END in text):
        return False
    block = render_block(items) if items else ""
    if START in text and END in text:
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1]
        new = head.rstrip("\n") + ("\n\n" + block if block else "") + tail
    else:
        new = text.rstrip("\n") + ("\n\n" + block + "\n" if block else "\n")
    if new == text:
        return False
    part.write_text(new, encoding="utf-8")
    return True
