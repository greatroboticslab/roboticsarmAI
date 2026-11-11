"""
[WIRED] Saved "photo positions" — named arm poses the arm can be sent back
to later so it can take a photo from exactly there (see the "Photo
Positions" panel on the Arm tab in main.py).

Each position is stored as JOINT values (j1, j2, z, j4) because that is
what the arm is actually commanded with (robot.movement.joint_to_joint_move),
so replaying a position that was saved from the arm's live pose puts the
arm back exactly where it was — no inverse-kinematics round trip. The
Cartesian x/y (robot frame — the same numbers the live X/Y readout on the
Arm tab shows) is stored alongside purely for display and for drawing the
marker on the plot.

This module is deliberately GUI-free (no tkinter, no robot): it only
validates, persists and edits the list, so it can be unit-tested on its
own. Persistence is a plain JSON file next to this module, same pattern
as hard_deck.py.
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import List, Optional

_PATH = os.path.join(os.path.dirname(__file__), "photo_positions.json")
_lock = threading.RLock()

# Same limits main.py's manual Move Joints panel enforces.
J1_LIMITS = (-85.0, 85.0)
J2_LIMITS = (-135.0, 135.0)
Z_LIMITS = (5.0, 245.0)
J4_LIMITS = (-358.0, 358.0)

# Arm link lengths used by main.Ikinematics (both 200 mm). Only used to
# derive a DISPLAY x/y for positions entered as joints.
_L1 = 200.0
_L2 = 200.0


def fk_xy(j1_deg: float, j2_deg: float) -> tuple:
    """Planar 2-link forward kinematics (inverse of main.Ikinematics).
    Returns the (x, y) in the robot frame."""
    t1 = math.radians(j1_deg)
    t12 = math.radians(j1_deg + j2_deg)
    return (_L1 * math.cos(t1) + _L2 * math.cos(t12),
            _L1 * math.sin(t1) + _L2 * math.sin(t12))


def validate_joints(j1: float, j2: float, z: float, j4: float) -> Optional[str]:
    """Returns an error message if any joint is out of range, else None."""
    for label, value, (lo, hi) in (("J1", j1, J1_LIMITS), ("J2", j2, J2_LIMITS),
                                    ("Z", z, Z_LIMITS), ("J4", j4, J4_LIMITS)):
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"{label} must be a number"
        if not (lo <= value <= hi):
            return f"{label} {value:g} is outside {lo:g} to {hi:g}"
    return None


def make_position(name: str, j1: float, j2: float, z: float, j4: float,
                  x: float = None, y: float = None, enabled: bool = True) -> dict:
    """Builds a validated position dict. Raises ValueError with a
    human-readable message if the name is blank or a joint is out of range.
    If x/y aren't supplied they're derived from the joints. `enabled` is
    the "include in the Photo Tour" toggle (a disabled position is kept
    and can still be gone to by hand, the tour just skips it)."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Position needs a name")
    j1, j2, z, j4 = (float(v) for v in (j1, j2, z, j4))
    err = validate_joints(j1, j2, z, j4)
    if err:
        raise ValueError(err)
    if x is None or y is None:
        x, y = fk_xy(j1, j2)
    return {"name": name, "j1": j1, "j2": j2, "z": z, "j4": j4,
            "x": float(x), "y": float(y), "enabled": bool(enabled)}


def _clean(entry) -> Optional[dict]:
    """Re-validates one record read from disk; None if it's unusable."""
    try:
        # Files saved before the toggle existed have no "enabled" key -> on.
        return make_position(entry["name"], entry["j1"], entry["j2"], entry["z"],
                             entry["j4"], entry.get("x"), entry.get("y"),
                             enabled=entry.get("enabled", True))
    except (KeyError, TypeError, ValueError):
        return None


def load() -> List[dict]:
    """Returns the saved positions, in order. Missing/corrupt file -> []."""
    with _lock:
        if not os.path.exists(_PATH):
            return []
        try:
            with open(_PATH, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    if not isinstance(raw, list):
        return []
    return [p for p in (_clean(e) for e in raw) if p is not None]


def save(positions: List[dict]) -> None:
    """Writes the whole list atomically (temp file + replace) so a crash
    mid-write can't leave a half-written file."""
    with _lock:
        tmp = _PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(positions, f, indent=2)
        os.replace(tmp, _PATH)


def unique_name(name: str, existing: List[dict]) -> str:
    """'shelf' -> 'shelf', then 'shelf_2', 'shelf_3' ... if already taken
    (case-insensitive), so names stay unambiguous in the list and in the
    photo names they end up in."""
    taken = {p["name"].lower() for p in existing}
    if name.lower() not in taken:
        return name
    n = 2
    while f"{name}_{n}".lower() in taken:
        n += 1
    return f"{name}_{n}"


def next_default_name(existing: List[dict]) -> str:
    """'pos_1', 'pos_2', ... first unused."""
    n = len(existing) + 1
    taken = {p["name"].lower() for p in existing}
    while f"pos_{n}" in taken:
        n += 1
    return f"pos_{n}"


def enabled_positions(positions: List[dict]) -> List[dict]:
    """The positions the Photo Tour should visit, in list order."""
    return [p for p in positions if p.get("enabled", True)]


def set_enabled(positions: List[dict], indices, enabled: bool) -> int:
    """Sets the tour toggle on positions[i] for each i in `indices`.
    Out-of-range indices are ignored. Returns how many were changed."""
    changed = 0
    for i in set(indices):
        if 0 <= i < len(positions) and positions[i].get("enabled", True) != enabled:
            positions[i]["enabled"] = enabled
            changed += 1
    return changed


def toggle(positions: List[dict], indices) -> Optional[bool]:
    """Toggle for a selection: if EVERY selected position is on they all go
    off, otherwise they all go on (so a mixed selection never flips
    unpredictably). Returns the new state, or None if nothing valid was
    selected."""
    valid = [i for i in set(indices) if 0 <= i < len(positions)]
    if not valid:
        return None
    new_state = not all(positions[i].get("enabled", True) for i in valid)
    set_enabled(positions, valid, new_state)
    return new_state
