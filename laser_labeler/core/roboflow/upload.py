"""Upload approved labels to existing Roboflow images (spec sections 9.3, 12).

Safety rules: dry run by default; only approved/edited sets; overwrite off unless
the plan says so; every real upload is appended to an audit log (never the key).
Only class names and boxes are sent. Metrics and links never leave the sidecar.

Routes (from docs, UNCONFIRMED LIVE; see docs/verified_facts.md):
- annotation: POST /dataset/{project}/annotate/{image_id}?name=...&overwrite=true
  with the raw COCO JSON as a text/plain body (a JSON body would be read as the
  {annotationFile, labelmap} wrapper).
- tags: POST /{workspace}/{project}/images/{image_id}/tags {"operation","tags"};
  fallback POST /{workspace}/images/{image_id}/metadata {"addTags": [...]}.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from core.models import ImageLabelSet
from core.roboflow.http import (RoboflowHTTPError, build_url, quote_seg, request_json,
                                scrub, with_backoff)
from core.schema.coco_export import EXPORTABLE_STATUSES, build_coco, class_names_for

ANNOTATION_NAME = "_annotations.coco.json"  # the extension tells Roboflow the format


@dataclass
class PlanItem:
    image_id: str
    image_name: str
    status: str
    boxes: list[tuple[str, tuple]]  # (class name, (x, y, w, h))
    tags: list[str]
    would_overwrite: bool | None  # None = unknown (remote labels not supplied)
    coco: dict = field(repr=False, default_factory=dict)

    @property
    def class_names(self) -> list[str]:
        return sorted({name for name, _ in self.boxes})


@dataclass
class UploadPlan:
    workspace: str
    project: str
    overwrite: bool = False
    add_to_dataset: bool | None = None  # None = leave Roboflow's default (true)
    items: list[PlanItem] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)  # image id -> reason

    def describe(self) -> str:
        """Exactly what would be sent: classes, boxes and tags per image."""
        ds = "Roboflow default (adds the image to the dataset)" if self.add_to_dataset is None \
            else ("yes" if self.add_to_dataset else "no")
        lines = [f"Upload plan for {self.workspace}/{self.project}",
                 f"  overwrite existing annotations: {'YES' if self.overwrite else 'no'}",
                 f"  add to dataset: {ds}",
                 f"  images to upload: {len(self.items)}, excluded: {len(self.excluded)}"]
        for n, item in enumerate(self.items, 1):
            lines.append(f"[{n}] {item.image_id}  {item.image_name}  ({item.status})")
            lines += [f"      box {name}  x,y,w,h = {box}" for name, box in item.boxes]
            lines.append(f"      tags: {', '.join(item.tags) or '(none)'}")
            lines.append("      existing annotation: " + _overwrite_note(item, self.overwrite))
        lines += [f"excluded {i}: {why}" for i, why in self.excluded.items()]
        return "\n".join(lines)


def _overwrite_note(item: PlanItem, overwrite: bool) -> str:
    if item.would_overwrite is None:
        return "not known (the server will refuse with 409 unless overwrite is on)"
    if not item.would_overwrite:
        return "none"
    return "yes, WILL BE REPLACED" if overwrite else "yes, will be SKIPPED (overwrite is off)"


def _tags_for(label_set: ImageLabelSet) -> list[str]:
    tags = []
    if label_set.pdf_path:
        tags.append(f"pdf:{Path(label_set.pdf_path).name}")
    if label_set.prompt_version:
        tags.append(f"prompt:{label_set.prompt_version}")
    return tags


def _plan_item(ls: ImageLabelSet, placeholder: str, remote: Mapping[str, Any]) -> PlanItem:
    if not (ls.image_width and ls.image_height):
        raise ValueError("image size is unknown")
    coco = build_coco([ls], placeholder)  # raises ValueError for an empty box
    labels = remote.get(ls.image_id)
    return PlanItem(ls.image_id, ls.image_name, ls.status, class_names_for(ls, placeholder),
                    _tags_for(ls), None if labels is None else bool(labels), coco)


def build_upload_plan(label_sets: Iterable[ImageLabelSet], *, workspace: str, project: str,
                      color_placeholder: str = "na", overwrite: bool = False,
                      remote_labels: Mapping[str, list[str] | None] | None = None,
                      add_to_dataset: bool | None = None) -> UploadPlan:
    """Pure: decide what would be sent. `remote_labels` (image id -> its current labels,
    from list_images) lets the plan say which images already have annotations."""
    plan = UploadPlan(workspace, project, overwrite, add_to_dataset)
    for ls in label_sets:
        if ls.status not in EXPORTABLE_STATUSES:
            plan.excluded[ls.image_id] = f"status is {ls.status}, only approved/edited are uploaded"
            continue
        try:
            plan.items.append(_plan_item(ls, color_placeholder, remote_labels or {}))
        except ValueError as exc:
            plan.excluded[ls.image_id] = f"cannot build annotation: {exc}"
    return plan


@dataclass
class UploadSummary:
    dry_run: bool = True
    uploaded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    tag_failed: dict[str, str] = field(default_factory=dict)  # annotation sent, tags not


def _send_annotation(plan: UploadPlan, item: PlanItem, key: str, opener: Callable,
                     timeout: float, sleep: Callable) -> None:
    ds = None if plan.add_to_dataset is None else str(plan.add_to_dataset).lower()
    url = build_url(f"dataset/{quote_seg(plan.project)}/annotate/{quote_seg(item.image_id)}", key,
                    name=ANNOTATION_NAME, overwrite="true" if plan.overwrite else None, addToDataset=ds)
    with_backoff(lambda: request_json("POST", url, body=json.dumps(item.coco), opener=opener,
                                      timeout=timeout, key_for_masking=key,
                                      headers={"Content-Type": "text/plain"}), sleep=sleep)


def add_tags(workspace: str, project: str, image_id: str, tags: list[str], api_key: str, *,
             opener: Callable = urllib.request.urlopen, timeout: float = 30,
             sleep: Callable[[float], None] = time.sleep) -> None:
    """Add tags to one image (documented tags route, then the metadata route)."""
    if not tags:
        return
    ws, pj, img = quote_seg(workspace), quote_seg(project), quote_seg(image_id)

    def post(path: str, body: dict) -> None:
        with_backoff(lambda: request_json("POST", build_url(path, api_key), body=body, opener=opener,
                                          timeout=timeout, key_for_masking=api_key), sleep=sleep)
    try:
        post(f"{ws}/{pj}/images/{img}/tags", {"operation": "add", "tags": tags})
    except RoboflowHTTPError as exc:
        if exc.status not in (400, 404, 405):
            raise
        post(f"{ws}/images/{img}/metadata", {"addTags": tags})


def _audit(log: Path, plan: UploadPlan, item: PlanItem, result: str, detail: str, key: str) -> None:
    row = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "workspace": plan.workspace, "project": plan.project, "image_id": item.image_id,
           "classes": item.class_names, "tags": item.tags, "overwrite": plan.overwrite,
           "result": result, "detail": scrub(detail, key)}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _why(exc: Exception, plan: UploadPlan, key: str) -> str:
    if getattr(exc, "status", None) == 409 and not plan.overwrite:
        return "HTTP 409: the image already has an annotation or a class is locked (overwrite is off)"
    return scrub(f"{type(exc).__name__}: {exc}", key)


def _upload_one(plan: UploadPlan, item: PlanItem, key: str, opener: Callable, timeout: float,
                sleep: Callable, summary: UploadSummary, log: Path) -> None:
    if item.status not in EXPORTABLE_STATUSES:
        summary.skipped[item.image_id] = f"status is {item.status}"
    elif item.would_overwrite and not plan.overwrite:
        summary.skipped[item.image_id] = "already has an annotation and overwrite is off"
    else:
        try:
            _send_annotation(plan, item, key, opener, timeout, sleep)
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop the batch
            summary.failed[item.image_id] = _why(exc, plan, key)
            return _audit(log, plan, item, "failed", summary.failed[item.image_id], key)
        summary.uploaded.append(item.image_id)
        try:
            add_tags(plan.workspace, plan.project, item.image_id, item.tags, key,
                     opener=opener, timeout=timeout, sleep=sleep)
        except Exception as exc:  # noqa: BLE001
            summary.tag_failed[item.image_id] = _why(exc, plan, key)
            return _audit(log, plan, item, "uploaded_tags_failed", summary.tag_failed[item.image_id], key)
        return _audit(log, plan, item, "uploaded", "", key)
    _audit(log, plan, item, "skipped", summary.skipped[item.image_id], key)


def upload_annotations(plan: UploadPlan, api_key: str, *, dry_run: bool = True,
                       opener: Callable = urllib.request.urlopen, audit_log: Path,
                       timeout: float = 30, sleep: Callable[[float], None] = time.sleep,
                       out: Callable[[str], None] = print) -> UploadSummary:
    """Send the plan. With dry_run (the default) it only prints the plan and sends nothing."""
    summary = UploadSummary(dry_run=dry_run)
    if dry_run:
        out(plan.describe())
        return summary
    for item in plan.items:
        _upload_one(plan, item, api_key, opener, timeout, sleep, summary, Path(audit_log))
    return summary
