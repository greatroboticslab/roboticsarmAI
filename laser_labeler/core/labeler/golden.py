"""Golden-set scoring and the provider comparison report (spec section 16).

Ground truth is a COCO JSON export of the hand-labeled images (class names follow
the obj_/las_ scheme). For each provider+mode run we measure box overlap with the
owner's boxes, attribute agreement on matched boxes, and the valid-reply rate.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from core.models import ImageLabelSet
from core.schema.class_names import (ClassNameError, ComponentClass, LaserClass, MergedClass, ObjectClass, decode_class)
from core.schema.coco_export import class_names_for
from core.tools.geometry import iou

MATCH_IOU = 0.5
_RF_SUFFIX = re.compile(r"^(.*?)\.rf\.[0-9a-zA-Z]+(\.\w+)$")
OBJECT_FIELDS = ("general_object", "material", "color", "incidence")


@dataclass
class TruthBox:
    class_name: str
    box: tuple[float, float, float, float]

    @property
    def decoded(self):
        return decode_class(self.class_name)


@dataclass
class FamilyScore:
    truth: int = 0
    predicted: int = 0
    matched: int = 0
    iou_sum: float = 0.0
    agree: dict[str, int] = field(default_factory=dict)  # attribute -> matched boxes that agree

    def add(self, other: "FamilyScore") -> None:
        self.truth += other.truth
        self.predicted += other.predicted
        self.matched += other.matched
        self.iou_sum += other.iou_sum
        for key, value in other.agree.items():
            self.agree[key] = self.agree.get(key, 0) + value

    @property
    def precision(self) -> float: return self.matched / self.predicted if self.predicted else 0.0
    @property
    def recall(self) -> float: return self.matched / self.truth if self.truth else 0.0
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0
    @property
    def mean_iou(self) -> float: return self.iou_sum / self.matched if self.matched else 0.0

    def agreement(self, name: str) -> float:
        return self.agree.get(name, 0) / self.matched if self.matched else 0.0


def normalize_name(file_name: str) -> str:
    """Roboflow exports add '.rf.<hash>' to file names; remove it so names match the originals."""
    match = _RF_SUFFIX.match(file_name)
    return (match.group(1) + match.group(2)) if match else file_name


def load_truth_coco(path: str | Path) -> dict[str, list[TruthBox]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data.get("categories", [])}
    files = {i["id"]: normalize_name(Path(i["file_name"]).name) for i in data.get("images", [])}
    truth: dict[str, list[TruthBox]] = {name: [] for name in files.values()}
    for ann in data.get("annotations", []):
        name = names.get(ann["category_id"], "")
        try:
            decode_class(name)
        except ClassNameError:
            continue  # the id-0 group category or a class outside our scheme
        truth[files[ann["image_id"]]].append(TruthBox(name, tuple(ann["bbox"])))
    return truth


def _match(truth: list[TruthBox], preds: list[tuple[str, tuple]]) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by IoU, best pairs first."""
    pairs = sorted(((iou(t.box, p[1]), ti, pi) for ti, t in enumerate(truth) for pi, p in enumerate(preds)),
                   reverse=True)
    used_t, used_p, out = set(), set(), []
    for score, ti, pi in pairs:
        if score < MATCH_IOU:
            break
        if ti not in used_t and pi not in used_p:
            used_t.add(ti), used_p.add(pi)
            out.append((ti, pi, score))
    return out


def _laser_part(decoded) -> str:
    if isinstance(decoded, LaserClass):
        return decoded.event
    return f"component{(decoded.component if isinstance(decoded, MergedClass) else decoded).number}"


def score_image(truth: Sequence[TruthBox], label_set: ImageLabelSet, color_placeholder: str = "na") -> dict[str, FamilyScore]:
    """Score one image. Failed or parked images simply have fewer predicted boxes."""
    try:
        predicted = class_names_for(label_set, color_placeholder)
    except ClassNameError:
        predicted = []  # a box with a value outside the scheme cannot be named: counts as missing
    scores = {"laser": FamilyScore(), "object": FamilyScore()}
    for family, kind in (("laser", (LaserClass, ComponentClass, MergedClass)), ("object", ObjectClass)):
        t = [b for b in truth if isinstance(b.decoded, kind)]
        p = [(n, box) for n, box in predicted if isinstance(decode_class(n), kind)]
        sc = scores[family]
        sc.truth, sc.predicted = len(t), len(p)
        for ti, pi, overlap in _match(t, p):
            sc.matched += 1
            sc.iou_sum += overlap
            want, got = t[ti].decoded, decode_class(p[pi][0])
            if family == "laser":  # an event name, or which part of the spot
                sc.agree["event"] = sc.agree.get("event", 0) + (_laser_part(want) == _laser_part(got))
                if isinstance(want, MergedClass) and isinstance(got, MergedClass):  # the object part of the name
                    for name in OBJECT_FIELDS:
                        sc.agree[name] = sc.agree.get(name, 0) + (getattr(want.object, name) == getattr(got.object, name))
                continue
            for name in OBJECT_FIELDS:
                sc.agree[name] = sc.agree.get(name, 0) + (getattr(want, name) == getattr(got, name))
    return scores


@dataclass
class RunScore:
    name: str
    images: int = 0
    failed: int = 0
    needs_input: int = 0
    clean_first_try: int = 0  # reply valid without any repair
    agent_ok: int = 0  # images that stayed in agent mode (tool calls worked)
    agent_requested: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    laser: FamilyScore = field(default_factory=FamilyScore)
    object: FamilyScore = field(default_factory=FamilyScore)

    @property
    def valid_reply_rate(self) -> float: return (self.images - self.failed) / self.images if self.images else 0.0
    @property
    def first_try_rate(self) -> float: return self.clean_first_try / self.images if self.images else 0.0


def _invalid_events(label_set: ImageLabelSet) -> int:
    try:
        entries = json.loads(Path(label_set.transcript_path).read_text(encoding="utf-8"))["entries"]
    except (OSError, ValueError, KeyError):
        return 0
    return sum(e["type"] in ("invalid_reply", "error") for e in entries)


def score_run(name: str, truth: dict[str, list[TruthBox]], label_sets: Sequence[ImageLabelSet],
              requested_mode: str = "pipeline", color_placeholder: str = "na") -> RunScore:
    run = RunScore(name, agent_requested=requested_mode == "agent")
    for ls in label_sets:
        run.images += 1
        run.failed += ls.status == "failed"
        run.needs_input += ls.status == "needs_input"
        run.clean_first_try += ls.status != "failed" and _invalid_events(ls) == 0
        run.agent_ok += ls.run_mode == "agent"
        scores = score_image(truth.get(normalize_name(ls.image_name), []), ls, color_placeholder)
        run.laser.add(scores["laser"])
        run.object.add(scores["object"])
    return run


def format_report(runs: Sequence[RunScore]) -> str:
    """Markdown comparison table, one column per run."""
    rows = [
        ("Images", lambda r: str(r.images)),
        ("Valid replies (not failed)", lambda r: f"{r.valid_reply_rate:.0%}"),
        ("Valid on first try (no repair)", lambda r: f"{r.first_try_rate:.0%}"),
        ("Stayed in agent mode", lambda r: f"{r.agent_ok}/{r.images}" if r.agent_requested else "n/a"),
        ("Parked for input", lambda r: str(r.needs_input)),
        ("Laser boxes: recall", lambda r: f"{r.laser.recall:.0%}"),
        ("Laser boxes: precision", lambda r: f"{r.laser.precision:.0%}"),
        ("Laser boxes: mean IoU", lambda r: f"{r.laser.mean_iou:.2f}"),
        ("Laser event agreement", lambda r: f"{r.laser.agreement('event'):.0%}"),
        ("Object boxes: recall", lambda r: f"{r.object.recall:.0%}"),
        ("Object boxes: precision", lambda r: f"{r.object.precision:.0%}"),
        ("Object boxes: mean IoU", lambda r: f"{r.object.mean_iou:.2f}"),
        *[(f"Object {f.replace('_', ' ')} agreement", (lambda r, f=f: f"{r.object.agreement(f):.0%}"))
          for f in OBJECT_FIELDS],
        ("Tokens in / out", lambda r: f"{r.input_tokens:,} / {r.output_tokens:,}"),
    ]
    lines = ["| Measure | " + " | ".join(r.name for r in runs) + " |",
             "| --- |" + " --- |" * len(runs)]
    lines += [f"| {label} | " + " | ".join(fn(r) for r in runs) + " |" for label, fn in rows]
    lines += ["", f"Boxes match when IoU >= {MATCH_IOU}. Agreement is measured on matched boxes only."]
    return "\n".join(lines)


def run_golden(name: str, image_paths: Sequence[Path], label_one: Callable[[Path], ImageLabelSet],
               truth: dict[str, list[TruthBox]], requested_mode: str, color_placeholder: str = "na") -> RunScore:
    """Label each golden image with `label_one` and score the results."""
    sets = [label_one(p) for p in image_paths]
    run = score_run(name, truth, sets, requested_mode, color_placeholder)
    for ls in sets:
        try:
            usage = json.loads(Path(ls.transcript_path).read_text(encoding="utf-8"))["usage"]
            run.input_tokens += usage["input_tokens"]
            run.output_tokens += usage["output_tokens"]
        except (OSError, ValueError, KeyError):
            pass
    return run
