"""Build the system prompt from named parts (spec section 7).

Parts are files in prompts/parts/, ordered by their numeric prefix. A name ending
in `.agent.md` or `.pipeline.md` is used only in that run mode. Placeholders
`{{taxonomy}}`, `{{table_rule}}` and `{{color_placeholder}}` are filled in.

Phase 3 version ids are a short hash of the text (`draft-ab12cd34`). Numbered
versions with a diff view arrive in Phase 8.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from core.schema.taxonomy import Taxonomy

DEFAULT_PARTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "parts"
MODES = ("agent", "pipeline")


@dataclass(frozen=True)
class PromptBuild:
    text: str
    version: str
    parts: tuple[str, ...]  # file names used, in order


def render_taxonomy(tax: Taxonomy, color_placeholder: str = "na") -> str:
    lines = [f"Objects: {', '.join(tax.objects)}"]
    lines += [
        f"Materials: {', '.join(tax.materials)}",
        f"Colors: {', '.join(tax.colors)} (use `{color_placeholder}` on grayscale images)",
        f"Laser events: {', '.join(tax.laser_events)}",
        f"Incidence: {', '.join(tax.incidence)}",
    ]
    return "\n".join(lines)


def _wanted(name: str, mode: str) -> bool:
    other = next(m for m in MODES if m != mode)
    return not name.endswith(f".{other}.md")


def list_parts(mode: str, parts_dir: Path = DEFAULT_PARTS_DIR) -> list[Path]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return [p for p in sorted(parts_dir.glob("*.md")) if _wanted(p.name, mode)]


def _saved_version(parts_dir: Path) -> str | None:
    """Version id (v0003) when building from a saved version folder, else None."""
    try:
        version = json.loads((parts_dir / "version.json").read_text(encoding="utf-8"))["version"]
    except (OSError, ValueError, KeyError):
        return None
    return version if isinstance(version, str) else None


def build_system_prompt(
    mode: str,
    taxonomy: Taxonomy,
    *,
    table_rule: str = "only_if_no_object_hit",
    color_placeholder: str = "na",
    parts_dir: Path = DEFAULT_PARTS_DIR,
) -> PromptBuild:
    parts = list_parts(mode, Path(parts_dir))
    if not parts:
        raise FileNotFoundError(f"no prompt parts found in {parts_dir}")
    values = {
        "{{taxonomy}}": render_taxonomy(taxonomy, color_placeholder),
        "{{table_rule}}": table_rule,
        "{{color_placeholder}}": color_placeholder,
    }
    text = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts)
    for key, value in values.items():
        text = text.replace(key, value)
    version = _saved_version(Path(parts_dir)) or "draft-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return PromptBuild(text, version, tuple(p.name for p in parts))
