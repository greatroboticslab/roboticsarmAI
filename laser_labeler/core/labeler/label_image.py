"""Label one image end to end: image in, sidecar JSON, transcript and CSV rows out (spec section 5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from core.camera import camera_id_for, load_image
from core.labeler.agent_mode import AgentModeFailed, run_agent
from core.labeler.context import build_context
from core.labeler.conversation import CostGuard
from core.review.reasons import question_kind
from core.labeler.finalize import RawResult, finalize
from core.labeler.parse_reply import ReplyParseError
from core.labeler.pipeline_mode import run_pipeline
from core.labeler.transcript import Transcript
from core.models import ImageLabelSet
from core.prompts.manager import build_system_prompt
from core.prompts.versions import prompt_dir_for_run
from core.providers.base import ProviderError, VisionProvider
from core.reports.manifest import update_manifest
from core.reports.measurements_csv import update_measurements
from core.reports.sidecar import safe_id, write_sidecar
from core.schema.taxonomy import Taxonomy


@dataclass
class LabelPaths:
    """Where one run writes its output (spec section 13: data/...)."""
    root: Path

    @property
    def sidecars(self) -> Path: return self.root / "sidecars"
    @property
    def transcripts(self) -> Path: return self.root / "transcripts"
    @property
    def measurements(self) -> Path: return self.root / "measurements.csv"
    @property
    def manifest(self) -> Path: return self.root / "manifest.csv"


def _run_mode(settings: dict, provider: VisionProvider, override: str | None) -> str:
    mode = override or settings["provider"]["run_mode"]
    return "pipeline" if mode == "agent" and not provider.supports_tools else mode


def _raw_result(ctx, provider, mode, settings, taxonomy, transcript, guard, prompt_dir) -> tuple[RawResult, str, str]:
    """Run the chosen mode; agent mode falls back to pipeline mode if the model misuses tools."""
    run, labeling = settings["run"], settings["labeling"]
    kwargs = dict(table_rule=labeling["table_rule"], color_placeholder=labeling["color_placeholder"],
                  parts_dir=prompt_dir)
    if mode == "agent":
        prompt = build_system_prompt("agent", taxonomy, **kwargs)
        try:
            raw = run_agent(ctx, provider, prompt.text, transcript, max_steps=run["max_agent_steps"],
                            max_repairs=run["max_repairs"], guard=guard)
            return raw, "agent", prompt.version
        except AgentModeFailed as exc:
            transcript.add_event("fallback", f"agent mode failed ({exc}); using pipeline mode")
    prompt = build_system_prompt("pipeline", taxonomy, **kwargs)
    raw = run_pipeline(ctx, provider, prompt.text, transcript, max_repairs=run["max_repairs"], guard=guard,
                       structured=bool(settings["provider"].get("structured_output")))
    return raw, "pipeline", prompt.version


def label_image_unsaved(
    path: str | Path,
    *,
    provider: VisionProvider,
    taxonomy: Taxonomy,
    settings: dict,
    image_id: str | None = None,
    tags: Sequence[str] = (),
    mode: str | None = None,
    guard: CostGuard | None = None,
    prompt_dir: str | Path | None = None,
    user_note: str = "",
    on_entry: Callable[[dict], None] | None = None,
) -> tuple[ImageLabelSet, Transcript]:
    """Label one image without writing anything. Provider failures give a status="failed" set instead of raising."""
    image_id = image_id or safe_id(Path(path).stem)
    transcript = Transcript(safe_id(image_id), listener=on_entry)
    labeling, run = settings["labeling"], settings["run"]
    image = load_image(path)
    camera = camera_id_for(path, tags, is_grayscale=image.is_grayscale, source=labeling["camera_source"],
                           tag_prefix=labeling["camera_tag_prefix"])
    ctx = build_context(image, taxonomy, max_image_px=settings["provider"]["max_image_px"],
                        color_placeholder=labeling["color_placeholder"], max_candidates=run["max_candidates"])
    ctx.user_note = user_note.strip()
    if ctx.user_note:
        transcript.add_event("reviewer_note", ctx.user_note)
    transcript.add_event("detection", ctx.detection.explain())  # what the app saw, before any model is asked
    try:
        raw, used_mode, version = _raw_result(ctx, provider, _run_mode(settings, provider, mode), settings,
                                              taxonomy, transcript, guard, Path(prompt_dir or prompt_dir_for_run()))
        result = finalize(ctx, raw, image_id=image_id, camera_id=camera, table_rule=labeling["table_rule"],
                          color_confidence=run["color_confidence"], question_confidence=run["question_confidence"],
                          scheme=labeling.get("scheme", "full"),
                          laser_object_fraction=float(labeling.get("laser_box_object_fraction", 0.0)))
        for q in result.questions:  # every question and what caused it, so the conversation explains the status
            transcript.add_event("question", f"[{question_kind(q)}] {q.text}")
        transcript.add_event("result", f"status {result.status}: {len(result.laser_boxes)} laser box(es), "
                                       f"{len(result.object_boxes)} object box(es), {len(result.questions)} question(s)")
    except (ProviderError, ReplyParseError) as exc:
        transcript.add_event("error", str(exc))
        used_mode, version = "", ""
        result = ImageLabelSet(image_id, Path(path).name, camera, image.is_grayscale, ctx.stats.background_level,
                               image.exposure, status="failed", image_width=image.width,
                               image_height=image.height, notes=f"failed: {exc}")
    result.provider, result.model = provider.name, provider.model
    result.prompt_version, result.run_mode = version, used_mode
    return result, transcript


def save_label_result(result: ImageLabelSet, transcript: Transcript, data_dir: str | Path) -> ImageLabelSet:
    """Write the transcript, sidecar, manifest row and measurement rows for one labeled image."""
    paths = LabelPaths(Path(data_dir))
    result.transcript_path = str(transcript.save(paths.transcripts))
    sidecar = write_sidecar(result, paths.sidecars)
    update_manifest(result, paths.manifest, str(sidecar))
    update_measurements(result, paths.measurements)
    return result


def label_image(path: str | Path, *, data_dir: str | Path, **kwargs) -> ImageLabelSet:
    """Label one image and save everything (spec section 5)."""
    result, transcript = label_image_unsaved(path, **kwargs)
    return save_label_result(result, transcript, data_dir)
