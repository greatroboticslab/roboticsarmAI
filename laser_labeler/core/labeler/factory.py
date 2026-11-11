"""Build the one function that labels a queue row. The CLI and the GUI both use it, so they behave the same.

It adds three things around `label_image`: the reviewer's re-run note, the spend guard, and the optional
hybrid escalation (a cloud model re-labels images the main model failed on or is unsure about).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import credentials
from core.labeler.conversation import CostGuard
from core.labeler.guard import TrackerGuard
from core.labeler.label_image import label_image_unsaved, save_label_result
from core.models import ImageLabelSet
from core.providers.base import VisionProvider
from core.providers.cost import CostTracker
from core.providers.registry import build_provider
from core.queue_db import QueueRow, overall_confidence
from core.review.service import take_rerun_note, take_rerun_points
from core.labeler.fit import FitMemory
from core.schema.class_names import ClassNameError
from core.schema.taxonomy import Taxonomy, TaxonomyError, add_entry

ESCALATION_PROVIDERS = ("gemini", "anthropic")  # cloud only: the OpenAI-compatible slot has one key for one server


@dataclass
class Escalation:
    provider: VisionProvider
    guard: CostGuard
    threshold: float


def make_tracker(block: dict, data_dir: Path) -> CostTracker:
    """Spend tracker from a settings block (the main provider or the escalation one). One shared daily store."""
    return CostTracker(block.get("price_in_per_mtok", 0), block.get("price_out_per_mtok", 0),
                       block.get("daily_cost_cap_usd"), Path(data_dir) / "spend.json")


def build_escalation(settings: dict, data_dir: Path) -> Escalation | None:
    """The escalation provider, or None when the setting is off. Raises ValueError/MissingCredentialError if misconfigured."""
    block = settings.get("escalation") or {}
    if not block.get("enabled"):
        return None
    kind = block.get("provider", "")
    if kind not in ESCALATION_PROVIDERS:
        raise ValueError(f"Escalation provider must be one of: {', '.join(ESCALATION_PROVIDERS)}")
    key = credentials.require_key(kind)
    provider = build_provider({**settings["provider"], "provider": kind, "model": block.get("model", "")}, key)
    priced = {**block, "daily_cost_cap_usd": settings["provider"].get("daily_cost_cap_usd")}  # one cap covers both
    return Escalation(provider, TrackerGuard(make_tracker(priced, data_dir)), float(block.get("confidence_threshold", 0.5)))


def needs_escalation(result: ImageLabelSet, threshold: float) -> str:
    """Why this result should go to the escalation model, or "" if it should not.

    An image with no laser found is not escalated: the model was never asked, so a bigger model cannot help."""
    if result.status == "failed":
        return "the first model failed"
    if result.laser_boxes and overall_confidence(result) < threshold:
        return f"confidence {overall_confidence(result):.2f} is below {threshold:.2f}"
    return ""


def build_label_fn(settings: dict, provider: VisionProvider, taxonomy: Taxonomy, data_dir: str | Path, *,
                   guard: CostGuard | None = None, mode: str | None = None, escalation: Escalation | None = None,
                   on_entry: Callable[[dict], None] | None = None,
                   prompt_dir: str | Path | None = None,
                   taxonomy_dir: str | Path | None = None) -> Callable[[QueueRow], ImageLabelSet]:
    data_dir = Path(data_dir)

    fit = FitMemory()  # the smaller request size that fitted the local context window, shared by this run's images

    def save_new_word(kind: str, value: str) -> None:
        """Taxonomy match off: write a new object or material to its YAML file (the in-memory lists already have it)."""
        if taxonomy_dir is None:
            return
        try:
            add_entry(taxonomy_dir, kind, value)
        except (TaxonomyError, ClassNameError):
            pass  # already there, or not a legal word: the label keeps it either way

    def label_fn(row: QueueRow) -> ImageLabelSet:
        common = dict(taxonomy=taxonomy, settings=settings, image_id=row.image_id, tags=row.tags, mode=mode,
                      prompt_dir=prompt_dir, on_entry=on_entry, user_note=take_rerun_note(data_dir, row.image_id),
                      hint_points=take_rerun_points(data_dir, row.image_id),
                      on_taxonomy_add=save_new_word)
        # a rate limit is raised, not recorded as a failed image: the run loop pauses and asks to resume
        result, transcript = label_image_unsaved(row.path, provider=provider, guard=guard, raise_on_limit=True, fit=fit, **common)
        reason = needs_escalation(result, escalation.threshold) if escalation else ""  # the second call never raises on a limit: the first answer is kept
        if reason:
            transcript.add_event("escalation", f"{reason}; asking {escalation.provider.name}/{escalation.provider.model}")
            second, second_transcript = label_image_unsaved(row.path, provider=escalation.provider,
                                                            guard=escalation.guard, **common)
            if second.status != "failed" or result.status == "failed":  # never trade a usable answer for a failed one
                second.notes = (f"escalated from {provider.name}/{provider.model}: {reason}. " + second.notes).strip()
                result, transcript = second, second_transcript
        return save_label_result(result, transcript, data_dir)

    return label_fn
