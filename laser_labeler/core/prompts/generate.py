"""Generate new prompt (spec section 7): the chosen AI drafts an updated prompt; the user approves it.

Nothing here changes a file until `apply_proposal` is called. The model may only rewrite existing prompt
parts or add new ones; it must keep every `{{placeholder}}` the part had, because the app fills those in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from core.labeler.conversation import CostGuard, call_model
from core.labeler.parse_reply import ReplyParseError, extract_json
from core.labeler.transcript import Transcript
from core.models import ImageLabelSet
from core.prompts.versions import active_version, save_version, set_active, taxonomy_snapshot
from core.providers.base import Message, VisionProvider
from core.schema.class_names import ClassNameError, encode_object_class, resolve_color
from core.schema.taxonomy import Taxonomy

PART_NAME = re.compile(r"^\d{2}_[a-z0-9_]+(\.agent|\.pipeline)?\.md$")
PLACEHOLDER = re.compile(r"\{\{[a-z_]+\}\}")
MAX_PART_CHARS = 40_000
KEEP_MARKERS = ("<!-- corrections:start -->", "<!-- corrections:end -->")

SYSTEM = """You maintain the system prompt of a tool that labels images of a laser hitting objects.
The prompt is a set of named text parts. You will be given the current parts, the taxonomy then and now, past corrections
made by the human reviewer, and a few approved labelings as examples. Update the prompt so the model will follow the new
taxonomy and avoid the mistakes the reviewer corrected.

Rules:
- Change as little as needed. Keep the voice, structure and every instruction that is still right.
- Keep every {{placeholder}} that a part already contains (for example {{taxonomy}}). The app fills them in.
- Never write the full taxonomy lists into the text: the {{taxonomy}} placeholder already does that.
- Keep any block between <!-- corrections:start --> and <!-- corrections:end --> exactly as it is.
- You may add guidance, examples or background where the new classes or corrections call for it.
Reply with ONLY one JSON object: {"summary": "one or two sentences on what you changed and why",
"parts": {"<file name>": "<the complete new text of that part>"}}. Include only parts you changed or added."""


class GenerateError(ValueError):
    """The model's proposal could not be used."""


@dataclass
class Proposal:
    summary: str
    parts: dict[str, str] = field(default_factory=dict)  # file name -> complete new text (changed or new parts only)


# ---------------------------------------------------------------- building the request
def read_parts(parts_dir: str | Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(parts_dir).glob("*.md"))}


def taxonomy_changes(old: dict | None, new: dict) -> str:
    """Plain-words list of what was added to or removed from the taxonomy."""
    if old is None:
        return "(no earlier taxonomy was stored for this version, so the changes are unknown)"
    lines: list[str] = []
    for key in ("materials", "colors", "laser_events", "incidence"):
        added = [v for v in new.get(key, []) if v not in old.get(key, [])]
        removed = [v for v in old.get(key, []) if v not in new.get(key, [])]
        if added:
            lines.append(f"{key}: added {', '.join(added)}")
        if removed:
            lines.append(f"{key}: removed {', '.join(removed)}")
    old_obj, new_obj = old.get("objects", {}), new.get("objects", {})
    for name, subs in new_obj.items():
        if name not in old_obj:
            lines.append(f"objects: new object {name} with sub-objects {', '.join(subs)}")
            continue
        fresh = [sub for sub in subs if sub not in old_obj[name]]
        if fresh:
            lines.append(f"objects: {name} gained sub-objects {', '.join(fresh)}")
    lines += [f"objects: removed object {name}" for name in old_obj if name not in new_obj]
    return "\n".join(lines) or "(the taxonomy has not changed)"


def render_example(ls: ImageLabelSet, placeholder: str = "na") -> str:
    parts = []
    for b in ls.laser_boxes:
        parts.append(f"  laser {b.class_name} box={list(b.box)} order={b.order}")
    for o in ls.object_boxes:
        try:
            name = encode_object_class(o.general_object, o.sub_object, o.material,
                                       resolve_color(o.color, ls.is_grayscale, placeholder), o.incidence)
        except ClassNameError:
            continue
        parts.append(f"  object {name} box={list(o.box)}")
    return f"{ls.image_name} ({'grayscale' if ls.is_grayscale else 'color'}):\n" + "\n".join(parts)


def build_messages(parts: dict[str, str], old_taxonomy: dict | None, new_taxonomy: Taxonomy,
                   corrections: Sequence[dict] = (), examples: Sequence[ImageLabelSet] = (),
                   placeholder: str = "na") -> list[Message]:
    sections = ["## Current prompt parts"]
    sections += [f"### {name}\n{text}" for name, text in parts.items()]
    sections.append("## Taxonomy changes since this prompt was written\n" + taxonomy_changes(old_taxonomy, new_taxonomy.to_dict()))
    sections.append("## Taxonomy now\n" + "\n".join(f"{k}: {v}" for k, v in new_taxonomy.to_dict().items()))
    if corrections:
        sections.append("## Corrections the reviewer made\n" + "\n".join(f"- {c['text']}" for c in corrections))
    if examples:
        sections.append("## Approved examples\n" + "\n".join(render_example(e, placeholder) for e in examples))
    return [Message("system", SYSTEM), Message("user", "\n\n".join(sections))]


# ---------------------------------------------------------------- reading the reply
def parse_proposal(text: str, current: dict[str, str]) -> Proposal:
    try:
        data = extract_json(text)
    except ReplyParseError as exc:
        raise GenerateError(str(exc)) from None
    parts = data.get("parts") if isinstance(data, dict) else None
    if not isinstance(parts, dict) or not parts:
        raise GenerateError('the reply needs a "parts" object with at least one part')
    clean: dict[str, str] = {}
    for name, body in parts.items():
        if not isinstance(name, str) or not PART_NAME.match(name):
            raise GenerateError(f"{name!r} is not a valid prompt part file name (like 05_questions.md)")
        if not isinstance(body, str) or not body.strip():
            raise GenerateError(f"part {name} is empty")
        if len(body) > MAX_PART_CHARS:
            raise GenerateError(f"part {name} is too long")
        old = current.get(name, "")
        lost = set(PLACEHOLDER.findall(old)) - set(PLACEHOLDER.findall(body))
        if lost:
            raise GenerateError(f"part {name} lost the placeholder(s) {', '.join(sorted(lost))}; keep them exactly")
        for marker in KEEP_MARKERS:
            if marker in old and marker not in body:
                raise GenerateError(f"part {name} lost the corrections block; keep it exactly")
        clean[name] = body.strip() + "\n"
    merged = {**current, **clean}
    if not any("{{taxonomy}}" in t for t in merged.values()):
        raise GenerateError("no part contains {{taxonomy}}, so the model would never see the allowed values")
    summary = data.get("summary")
    return Proposal(summary.strip() if isinstance(summary, str) else "", clean)


def generate_new_prompt(provider: VisionProvider, parts_dir: str | Path, old_taxonomy: dict | None,
                        new_taxonomy: Taxonomy, *, corrections: Sequence[dict] = (),
                        examples: Sequence[ImageLabelSet] = (), placeholder: str = "na",
                        guard: CostGuard | None = None, max_repairs: int = 2) -> Proposal:
    """Ask the model for an updated prompt. Invalid replies are sent back for repair, like labeling replies."""
    current = read_parts(parts_dir)
    if not current:
        raise GenerateError(f"no prompt parts found in {parts_dir}")
    messages = build_messages(current, old_taxonomy, new_taxonomy, corrections, examples, placeholder)
    transcript = Transcript("prompt-generation")
    reply = call_model(provider, messages, transcript, None, guard)
    for attempt in range(max_repairs + 1):
        try:
            return parse_proposal(reply.text, current)
        except GenerateError as exc:
            if attempt == max_repairs:
                raise
            messages.append(Message("user", f"That reply could not be used: {exc}. Reply again with ONLY the corrected JSON object."))
            reply = call_model(provider, messages, transcript, None, guard)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- showing and applying it
def proposal_pairs(parts_dir: str | Path, proposal: Proposal) -> list[tuple[str, str, str]]:
    """(file name, current text, proposed text) for every part the proposal touches, for the side-by-side view."""
    current = read_parts(parts_dir)
    return [(name, current.get(name, ""), new) for name, new in sorted(proposal.parts.items())
            if current.get(name, "") != new]


def apply_proposal(proposal: Proposal, parts_dir: str | Path, versions_dir: str | Path, taxonomy: Taxonomy,
                   note: str = "") -> str:
    """Write the approved parts to the draft, save them as a new numbered version and make it the active one."""
    parts_dir = Path(parts_dir)
    for name, body in proposal.parts.items():
        if not PART_NAME.match(name):
            raise GenerateError(f"{name!r} is not a valid prompt part file name")
        (parts_dir / name).write_text(body, encoding="utf-8")
    version = save_version(parts_dir, versions_dir, note or proposal.summary or "generated", taxonomy=taxonomy.to_dict())
    set_active(versions_dir, version)
    return version


def old_taxonomy_for_active(versions_dir: str | Path) -> dict | None:
    active = active_version(versions_dir)
    return taxonomy_snapshot(versions_dir, active) if active else None
