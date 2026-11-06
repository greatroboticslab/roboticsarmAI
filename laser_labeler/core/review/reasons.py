"""Plain-words reasons an image needs input, from the questions the labeler raised (no GUI code)."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from core.models import ImageLabelSet, Question

NO_LASER = "no laser found"
LOW_CONFIDENCE = "low confidence"
COLOR = "color unclear"
TAXONOMY = "value not in your lists"
INCIDENCE = "incidence not valid"
MODEL = "the model asked"


def question_kind(q: Question) -> str:
    """A short label for the cause of one question, from its id (ids are set by the labeler)."""
    qid = q.id or ""
    if qid == "q_no_laser" or q.field_name == "hit_point":
        return NO_LASER
    if qid.startswith("ql_"):
        return LOW_CONFIDENCE
    if qid.startswith("qc_"):
        return COLOR
    if qid.startswith("qt_"):
        return TAXONOMY
    if qid.startswith("qi_"):
        return INCIDENCE
    return MODEL


def open_questions(label_set: ImageLabelSet) -> list[Question]:
    return [q for q in label_set.questions if q.answer is None]


def reason_line(label_set: ImageLabelSet) -> str:
    """One line for a list row, for example 'no laser found' or 'low confidence, color unclear'."""
    kinds = []
    for q in open_questions(label_set):
        kind = question_kind(q)
        if kind not in kinds:
            kinds.append(kind)
    return ", ".join(kinds)


def summarize(label_sets: Iterable[ImageLabelSet]) -> Counter:
    """How many images wait on each kind of question (an image with two kinds counts once in each)."""
    counts: Counter = Counter()
    for ls in label_sets:
        counts.update({question_kind(q) for q in open_questions(ls)})
    return counts


def describe_summary(counts: Counter) -> str:
    return ", ".join(f"{n} x {kind}" for kind, n in counts.most_common())
