"""Which images the Select images tab shows and ticks (no GUI code)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

LABELED, UNLABELED, UNKNOWN = "labeled", "unlabeled", "unknown"


@dataclass
class Entry:
    key: str  # image id (Roboflow id, or the id the queue will use for a local file)
    name: str
    path: str | None = None  # local file, when there is one
    split: str | None = None
    tags: list[str] = field(default_factory=list)
    labels: list[str] | None = None  # None = not known (local files, or Roboflow did not say)
    remote: object = None  # the RemoteImage for Roboflow entries


def label_state(entry: Entry) -> str:
    """Local files and images whose labels Roboflow did not report are `unknown` (the user is asked)."""
    if entry.labels is None:
        return UNKNOWN
    return LABELED if entry.labels else UNLABELED


def filter_entries(entries: Iterable[Entry], *, only_unlabeled: bool = False, split: str = "", tag: str = "",
                   text: str = "", queued: Iterable[str] = (), hide_queued: bool = True) -> list[Entry]:
    """Entries to show. "Unlabeled only" keeps unlabeled images and ones of unknown state (never silently drops them)."""
    queued_ids, needle = set(queued), text.strip().lower()
    out = []
    for e in entries:
        if hide_queued and e.key in queued_ids:
            continue
        if only_unlabeled and label_state(e) == LABELED:
            continue
        if split and e.split != split:
            continue
        if tag and tag not in e.tags:
            continue
        if needle and needle not in e.name.lower() and needle not in e.key.lower():
            continue
        out.append(e)
    return out


def next_unticked(shown: Sequence[Entry], ticked: Iterable[str], count: int) -> list[str]:
    """The next `count` shown images that are not ticked yet, in the order shown (the chunk picker)."""
    have = set(ticked)
    return [e.key for e in shown if e.key not in have][:max(count, 0)]


def split_options(entries: Iterable[Entry]) -> list[str]:
    return sorted({e.split for e in entries if e.split})


def tag_options(entries: Iterable[Entry]) -> list[str]:
    return sorted({t for e in entries for t in e.tags})
