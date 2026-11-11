"""Shows the AI conversation: live while a run is going, or loaded from a saved transcript."""
from __future__ import annotations

import html
import json
from pathlib import Path

from PyQt6.QtWidgets import QTextBrowser, QWidget

from core.labeler.transcript import load_transcript

SYSTEM_PREVIEW = 400  # characters of the system prompt shown (it is long and the same every time)
TEXT_LIMIT = 4000
ROLE_COLORS = {"system": "#7a7f87", "user": "#1f6feb", "assistant": "#1a7f4b", "tool": "#9a6700"}
EVENT_COLORS = {"answer": "#8250df", "reviewer_note": "#8250df", "reviewer_marks": "#8250df", "error": "#cf222e", "fallback": "#bc4c00",
                "invalid_reply": "#bc4c00", "escalation": "#bc4c00", "detection": "#0a7ea4",
                "question": "#bc4c00", "result": "#1a7f4b"}


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters)"


class ConversationViewer(QTextBrowser):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOpenLinks(False)
        self._image_dir: Path | None = None
        self.clear_view()

    def clear_view(self, placeholder: str = "Nothing to show yet.") -> None:
        self.clear()
        self._image_dir = None
        self._empty = True
        self.setPlaceholderText(placeholder)

    def append_entry(self, entry: dict, image_dir: str | Path | None = None) -> None:
        directory = Path(image_dir) if image_dir else self._image_dir
        self.append(self._entry_html(entry, directory))
        self._empty = False
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def load(self, transcript_path: str | Path | None) -> bool:
        """Show a saved transcript with its crop images. False if it cannot be read."""
        self.clear_view("This image has no saved conversation.")
        if not transcript_path:
            return False
        try:
            data = load_transcript(transcript_path)
        except (OSError, ValueError):
            return False
        folder = Path(transcript_path).with_suffix("")
        self._image_dir = folder if folder.is_dir() else None
        usage = data.get("usage") or {}
        self.append(f"<p style='color:#7a7f87'>Tokens: {usage.get('input_tokens', 0)} in, {usage.get('output_tokens', 0)} out</p>")
        for entry in data.get("entries", []):
            self.append_entry(entry)
        self.verticalScrollBar().setValue(0)
        return True

    def _entry_html(self, entry: dict, image_dir: Path | None) -> str:
        stamp = html.escape(str(entry.get("time", ""))[11:19])
        if entry.get("type") != "message":
            kind = str(entry.get("type", "note"))
            color = EVENT_COLORS.get(kind, "#7a7f87")
            return (f"<p><span style='color:{color}'><b>{stamp} {html.escape(kind.replace('_', ' ').upper())}</b></span> "
                    f"{html.escape(_clip(str(entry.get('text', '')), TEXT_LIMIT))}</p>")
        role = entry.get("role", "?")
        text = str(entry.get("text") or "")
        if role == "system":
            text = _clip(text, SYSTEM_PREVIEW)
        else:
            text = _clip(text, TEXT_LIMIT)
        label = role.upper() + (f" ({html.escape(str(entry['tool_name']))})" if entry.get("tool_name") else "")
        parts = [f"<p><span style='color:{ROLE_COLORS.get(role, '#444')}'><b>{stamp} {label}</b></span><br>"
                 f"<span style='white-space:pre-wrap'>{html.escape(text)}</span>"]
        for call in entry.get("tool_calls") or []:
            parts.append(f"<br><b>tool call</b> {html.escape(str(call.get('name')))} "
                         f"<code>{html.escape(json.dumps(call.get('arguments')))}</code>")
        for name in entry.get("images") or []:
            path = image_dir / name if image_dir else None
            if path is not None and path.is_file():
                parts.append(f"<br><img src='{path.as_uri()}' width='220'>")
            else:
                parts.append(f"<br><i>[image {html.escape(str(name))}]</i>")
        parts.append("</p>")
        return "".join(parts)
