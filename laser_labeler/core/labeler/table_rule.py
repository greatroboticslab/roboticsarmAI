"""Table rule (build spec section 2).

Default reading, confirmed with the owner as a stated interpretation: label the
table only if NO object in the image has a laser hit on it. If any non-table
object is hit, drop every table box. Kept as a config flag because the owner's
original wording was ambiguous.
"""
from __future__ import annotations

from core.models import ObjectBox

TABLE_OBJECT = "table"

ONLY_IF_NO_OBJECT_HIT = "only_if_no_object_hit"
ALWAYS_LABEL = "always_label"
NEVER_LABEL = "never_label"
RULES = (ONLY_IF_NO_OBJECT_HIT, ALWAYS_LABEL, NEVER_LABEL)


def is_table(box: ObjectBox) -> bool:
    return box.general_object == TABLE_OBJECT


def _is_hit(box: ObjectBox) -> bool:
    return bool(box.laser_ids)


def apply_table_rule(
    boxes: list[ObjectBox], rule: str = ONLY_IF_NO_OBJECT_HIT
) -> list[ObjectBox]:
    """Return the object boxes that survive the rule, in their original order."""
    if rule not in RULES:
        raise ValueError(f"unknown table rule {rule!r}; expected one of {RULES}")
    if rule == ALWAYS_LABEL:
        return list(boxes)
    if rule == NEVER_LABEL:
        return [b for b in boxes if not is_table(b)]
    other_object_hit = any(_is_hit(b) for b in boxes if not is_table(b))
    if other_object_hit:
        return [b for b in boxes if not is_table(b)]
    return list(boxes)
