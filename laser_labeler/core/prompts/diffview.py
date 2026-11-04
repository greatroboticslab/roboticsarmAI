"""Line-by-line alignment of two texts for a side-by-side diff (no GUI code)."""
from __future__ import annotations

import difflib

Row = tuple[str, str, str]  # (left text, right text, tag)  tag: equal | replace | delete | insert


def align_diff(old: str, new: str) -> list[Row]:
    """Rows to show side by side. A removed line has an empty right side, an added line an empty left side,
    and a changed block is paired line by line (the shorter side is padded with empty rows)."""
    a, b = old.splitlines(), new.splitlines()
    rows: list[Row] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            rows += [(line, line, "equal") for line in a[i1:i2]]
        elif tag == "delete":
            rows += [(line, "", "delete") for line in a[i1:i2]]
        elif tag == "insert":
            rows += [("", line, "insert") for line in b[j1:j2]]
        else:
            left, right = a[i1:i2], b[j1:j2]
            for n in range(max(len(left), len(right))):
                rows.append((left[n] if n < len(left) else "", right[n] if n < len(right) else "", "replace"))
    return rows


def changed_count(rows: list[Row]) -> int:
    return sum(tag != "equal" for _, _, tag in rows)
