"""Describe what a reviewer changed between two label sets, in plain words.

The same list feeds the PDF addendum, the corrections file (examples for the prompt) and the
review screen's "what changed" line.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.models import ImageLabelSet

OBJECT_FIELDS = ("general_object", "sub_object", "material", "color", "incidence", "finish")
def _name(box) -> str:
    return box.component or box.kind


LASER_FIELDS = ("kind", "component", "order", "from_id")


@dataclass(frozen=True)
class Change:
    kind: str  # attribute | box | added | removed | metric
    box_id: str
    field: str = ""
    before: str = ""
    after: str = ""
    context: str = ""  # what the box was, for examples

    def describe(self) -> str:
        who = f"{self.box_id}" + (f" ({self.context})" if self.context else "")
        if self.kind == "attribute":
            return f"{who}: {self.field} {self.before or '(none)'} -> {self.after or '(none)'}"
        if self.kind == "box":
            return f"{who}: box moved/resized {self.before} -> {self.after}"
        if self.kind == "metric":
            return f"{who}: measurement {self.field} {self.before} -> {self.after}"
        if self.kind == "added":
            return f"{who}: box added ({self.after})"
        return f"{who}: box removed ({self.before})"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "box_id": self.box_id, "field": self.field, "before": self.before,
                "after": self.after, "context": self.context}


def _obj_context(o) -> str:
    return f"{o.general_object}/{o.sub_object} {o.material} {o.color}"


def _same(a, b) -> bool:
    return a == b


def diff_label_sets(before: ImageLabelSet, after: ImageLabelSet) -> list[Change]:
    """Every difference a reviewer could have made: attributes, boxes, added/removed boxes, edited numbers."""
    changes: list[Change] = []
    old_o = {o.id: o for o in before.object_boxes}
    new_o = {o.id: o for o in after.object_boxes}
    for oid, new in new_o.items():
        old = old_o.get(oid)
        if old is None:
            changes.append(Change("added", oid, after=f"{_obj_context(new)} at {list(new.box)}"))
            continue
        for name in OBJECT_FIELDS:
            if not _same(getattr(old, name), getattr(new, name)):
                changes.append(Change("attribute", oid, name, str(getattr(old, name) or ""),
                                      str(getattr(new, name) or ""), _obj_context(old)))
        if tuple(old.box) != tuple(new.box):
            changes.append(Change("box", oid, before=str(list(old.box)), after=str(list(new.box)),
                                  context=_obj_context(new)))
    changes += [Change("removed", oid, before=f"{_obj_context(o)} at {list(o.box)}") for oid, o in old_o.items()
                if oid not in new_o]
    old_l = {b.id: b for b in before.laser_boxes}
    new_l = {b.id: b for b in after.laser_boxes}
    for lid, new in new_l.items():
        old = old_l.get(lid)
        if old is None:
            changes.append(Change("added", lid, after=f"laser {_name(new)} at {list(new.box)}"))
            continue
        for name in LASER_FIELDS:
            if not _same(getattr(old, name), getattr(new, name)):
                changes.append(Change("attribute", lid, name, str(getattr(old, name) or ""),
                                      str(getattr(new, name) or ""), f"laser {_name(old)}"))
        if tuple(old.box) != tuple(new.box):
            changes.append(Change("box", lid, before=str(list(old.box)), after=str(list(new.box)),
                                  context=f"laser {_name(new)}"))
        for key in sorted(set(old.metrics) | set(new.metrics)):
            a, b = old.metrics.get(key), new.metrics.get(key)
            if a != b and tuple(old.box) == tuple(new.box):  # a moved box re-measures: only report hand edits
                changes.append(Change("metric", lid, key, f"{a:.4g}" if isinstance(a, (int, float)) else str(a),
                                      f"{b:.4g}" if isinstance(b, (int, float)) else str(b), f"laser {_name(new)}"))
    changes += [Change("removed", lid, before=f"laser {_name(b)} at {list(b.box)}") for lid, b in old_l.items()
                if lid not in new_l]
    return changes
