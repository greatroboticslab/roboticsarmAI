"""Record the full AI conversation for one image (feeds the PDF in Phase 7).

Entries are saved as JSON; images go in a folder beside it and are referenced by
relative path. API keys are never part of a message, so nothing needs scrubbing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from core.providers.base import Message, Usage


@dataclass
class Transcript:
    image_id: str
    entries: list[dict] = field(default_factory=list)
    _images: list[tuple[str, bytes]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    listener: Callable[[dict], None] | None = None  # called with each new entry (live view)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _record(self, entry: dict) -> None:
        self.entries.append(entry)
        if self.listener:
            try:
                self.listener(entry)
            except Exception:  # a broken viewer must never break labeling
                pass

    def add_message(self, message: Message) -> None:
        names = []
        for part in message.images:
            name = f"img_{len(self._images) + 1:03d}.{'jpg' if 'jpeg' in part.mime else 'png'}"
            self._images.append((name, part.data))
            names.append(name)
        self._record({
            "time": self._now(), "type": "message", "role": message.role, "text": message.text,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls],
            "tool_call_id": message.tool_call_id, "tool_name": message.tool_name, "images": names,
        })

    def add_event(self, kind: str, text: str) -> None:
        """Non-message notes: fallbacks, repairs, parked questions, user answers."""
        self._record({"time": self._now(), "type": kind, "text": text})

    def add_usage(self, usage: Usage) -> None:
        self.usage.input_tokens += usage.input_tokens
        self.usage.output_tokens += usage.output_tokens

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        folder = directory / self.image_id
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in self._images:
            (folder / name).write_bytes(data)
        path = directory / f"{self.image_id}.json"
        payload = {"image_id": self.image_id, "usage": vars(self.usage), "entries": self.entries}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def load_transcript(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_event(path: str | Path, kind: str, text: str) -> None:
    """Add a note (for example the user's answer to a question) to a saved transcript."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["entries"].append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "type": kind, "text": text})
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
