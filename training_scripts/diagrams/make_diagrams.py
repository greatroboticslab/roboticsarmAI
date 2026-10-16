#!/usr/bin/env python3
"""
make_diagrams.py

Redraws the four evaluation diagrams used in the READMEs, in very large type:

    01_evaluation_pipeline.png      how one image is evaluated, start to finish
    02_label_levels.png             one class name scored four ways: exact class / object / material / color
    03_found_but_wrong_label.png    a box WAS found on a real object -- correct, or which part of the label was wrong
    04_missed_and_extra_boxes.png   real objects the model missed, and boxes where nothing is

Edit the text or the numbers below and re-run to regenerate:
    python make_diagrams.py            # writes the PNGs next to this script
    python make_diagrams.py --scale 1.2   # even bigger text (boxes may need resizing)

Requires matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- palette (also readable in print / for colour-blind viewers: every state also has a word or a symbol)
INK = "#14181f"
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f5fbf", "#d9480f", "#1b8a3a", "#c92a2a", "#6741d9", "#495057"
LIGHT = {BLUE: "#dbe7fb", ORANGE: "#fde3d3", GREEN: "#d6f1dd", RED: "#fadada", PURPLE: "#e6dffa", GREY: "#e9ecef"}
FIGSIZE = (16, 9)
SCALE = 1.0


def canvas(title: str, subtitle: str | None = None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    ax.text(8, 8.55, title, ha="center", va="center", fontsize=34 * SCALE, fontweight="bold", color=INK)
    if subtitle:
        ax.text(8, 7.95, subtitle, ha="center", va="center", fontsize=24 * SCALE, color=GREY)
    return fig, ax


def box(ax, cx, cy, w, h, text, color, fs=26, filled=True, bold=True, tcolor=None):
    face = color if filled else LIGHT[color]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=face, ec=color, lw=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs * SCALE,
            fontweight="bold" if bold else "normal", color=tcolor or ("white" if filled else INK),
            linespacing=1.15)


def arrow(ax, p1, p2, label=None, color=INK, fs=24, label_offset=(0, 0.28), lw=4, elbow=None):
    pts = [p1] + ([elbow] if elbow else []) + [p2]
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="round", zorder=1)
    a, b = pts[-2], pts[-1]
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=34), zorder=2)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + label_offset[0], (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs * SCALE, fontweight="bold", color=color)


def badge(ax, x, y, n):
    ax.add_patch(plt.Circle((x, y), 0.3, fc=INK, ec="white", lw=3, zorder=5))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=24 * SCALE, fontweight="bold", color="white", zorder=6)


# --------------------------------------------------------------------------- #
def d1(out: Path):
    fig, ax = canvas("How one image is evaluated")
    W, H = 4.3, 1.75
    X = [2.5, 8, 13.5]
    Y = [6.5, 3.95, 1.4]
    box(ax, X[0], Y[0], W, H, "PHOTO\nfrom the dataset", GREY)
    box(ax, X[1], Y[0], W, H, "MODEL\nYOLO or RF-DETR", BLUE)
    box(ax, X[2], Y[0], W, H, "PREDICTIONS\nbox + class + score", BLUE, fs=24)
    arrow(ax, (X[0] + W / 2, Y[0]), (X[1] - W / 2, Y[0]))
    arrow(ax, (X[1] + W / 2, Y[0]), (X[2] - W / 2, Y[0]))
    box(ax, X[0], Y[1], W, H, "KEEP BOXES\nscore ≥ 0.3", ORANGE)
    box(ax, X[1], Y[1], W, H, "MATCH TO TRUTH\noverlap (IoU) ≥ 0.5", ORANGE, fs=24)
    box(ax, X[2], Y[1], W, H, "COMPARE\nthe class names", ORANGE)
    # elbow from PREDICTIONS down to KEEP BOXES
    ax.plot([X[2], X[2]], [Y[0] - H / 2, 5.22], color=INK, lw=4, zorder=1)
    ax.plot([X[2], X[0]], [5.22, 5.22], color=INK, lw=4, zorder=1)
    ax.annotate("", xy=(X[0], Y[1] + H / 2), xytext=(X[0], 5.22),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=4, mutation_scale=34))
    arrow(ax, (X[0] + W / 2, Y[1]), (X[1] - W / 2, Y[1]))
    arrow(ax, (X[1] + W / 2, Y[1]), (X[2] - W / 2, Y[1]))
    box(ax, X[1], Y[2], W, H, "TRUE LABELS\nObject + Material\n+ Color", GREEN, fs=24)
    arrow(ax, (X[1], Y[2] + H / 2), (X[1], Y[1] - H / 2))
    box(ax, X[2], Y[2], W, H, "SORT + SCORE\ncorrect/  or  failure/", PURPLE, fs=25)
    arrow(ax, (X[2], Y[1] - H / 2), (X[2], Y[2] + H / 2))
    box(ax, X[0], Y[2], W, H, "0.3 and 0.5\ncan be changed", GREY, filled=False, fs=24, bold=False)
    for n, (x, y) in enumerate([(X[0], Y[0]), (X[1], Y[0]), (X[2], Y[0]), (X[0], Y[1]), (X[1], Y[1]), (X[2], Y[1])], 1):
        badge(ax, x - W / 2 + 0.05, y + H / 2 - 0.05, n)
    fig.savefig(out / "01_evaluation_pipeline.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def d2(out: Path):
    fig, ax = canvas("One class name, scored four ways")
    ax.text(1.0, 7.6, "TRUE:", ha="left", va="center", fontsize=28 * SCALE, fontweight="bold", color=INK)
    box(ax, 5.2, 7.6, 3.9, 0.85, "ballpoint pen", BLUE, fs=26)
    ax.text(7.45, 7.6, "+", ha="center", va="center", fontsize=30, fontweight="bold")
    box(ax, 9.45, 7.6, 3.0, 0.85, "plastic", ORANGE, fs=26)
    ax.text(11.2, 7.6, "+", ha="center", va="center", fontsize=30, fontweight="bold")
    box(ax, 13.0, 7.6, 2.6, 0.85, "red", GREEN, fs=26)
    ax.text(5.2, 6.78, "OBJECT", ha="center", fontsize=20, color=BLUE, fontweight="bold")
    ax.text(9.45, 6.78, "MATERIAL", ha="center", fontsize=20, color=ORANGE, fontweight="bold")
    ax.text(13.0, 6.78, "COLOR", ha="center", fontsize=20, color=GREEN, fontweight="bold")

    cols = [("MODEL SAID", 4.0, None), ("EXACT\nCLASS", 8.9, None), ("OBJECT", 10.9, BLUE), ("MATERIAL", 12.9, ORANGE), ("COLOR", 14.9, GREEN)]
    top = 6.1
    for name, x, c in cols:
        ax.text(x, top, name, ha="center", va="center", fontsize=22 * SCALE, fontweight="bold", color=c or INK)
    rows = [
        ("ballpoint pen · plastic · red", [1, 1, 1, 1]),
        ("ballpoint pen · rubber · red", [0, 1, 0, 1]),
        ("ballpoint pen · plastic · blue", [0, 1, 1, 0]),
        ("gel pen · plastic · red", [0, 0, 1, 1]),
        ("rubber duck · rubber · yellow", [0, 0, 0, 0]),
    ]
    rh = 0.86
    for i, (label, marks) in enumerate(rows):
        y = 5.15 - i * rh
        ax.add_patch(FancyBboxPatch((0.25, y - rh / 2 + 0.05), 15.5, rh - 0.1, boxstyle="round,pad=0.01,rounding_size=0.1",
                                    fc="#f4f5f7", ec="none", zorder=0))
        ax.text(0.5, y, label, ha="left", va="center", fontsize=22 * SCALE, fontweight="bold", color=INK)
        for (name, x, c), m in zip(cols[1:], marks):
            ok = bool(m)
            ax.add_patch(FancyBboxPatch((x - 0.78, y - 0.34), 1.56, 0.68, boxstyle="round,pad=0.01,rounding_size=0.12",
                                        fc=GREEN if ok else RED, ec="none"))
            ax.text(x, y - 0.02, "✓" if ok else "✗", ha="center", va="center", fontsize=30, fontweight="bold", color="white")
    ax.text(8, 0.78, "Overall score = EXACT CLASS only  (the strictest)", ha="center", fontsize=24 * SCALE, fontweight="bold", color=INK)
    ax.text(8, 0.28, "Composite = 0.6 × Object + 0.3 × Material + 0.1 × Color", ha="center", fontsize=24 * SCALE, fontweight="bold", color=PURPLE)
    fig.savefig(out / "02_label_levels.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def d3(out: Path):
    fig, ax = canvas("A box was found on a real object", "Was the label right? If not, which part was wrong?")
    qx, qw, qh = 2.5, 4.2, 1.35
    ox, ow = 11.3, 6.6
    box(ax, qx, 6.55, qw, qh, "Exact class\nright?", GREY, fs=27)
    box(ax, ox, 6.55, ow, 1.1, "CORRECT", GREEN, fs=30)
    arrow(ax, (qx + qw / 2, 6.55), (ox - ow / 2, 6.55), "YES", GREEN, fs=26, label_offset=(0, 0.38))
    box(ax, qx, 4.45, qw, qh, "Same object\nname?", GREY, fs=27)
    arrow(ax, (qx, 6.55 - qh / 2), (qx, 4.45 + qh / 2), "NO", RED, fs=26, label_offset=(0.5, 0))
    box(ax, ox, 4.45, ow, 1.45, "WRONG OBJECT\npen called a hammer", RED, fs=28)
    arrow(ax, (qx + qw / 2, 4.45), (ox - ow / 2, 4.45), "NO", RED, fs=26, label_offset=(0, 0.38))
    box(ax, qx, 1.75, qw, 1.2, "What differs?", GREY, fs=27)
    arrow(ax, (qx, 4.45 - qh / 2), (qx, 1.75 + 0.6), "YES", GREEN, fs=26, label_offset=(0.55, 0))
    ys = (2.95, 1.75, 0.55)
    for yy, t in zip(ys, ("WRONG MATERIAL", "WRONG COLOR", "BOTH WRONG")):
        box(ax, ox, yy, ow, 0.95, t, ORANGE, fs=27)
        ax.annotate("", xy=(ox - ow / 2, yy), xytext=(6.6, yy),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=4, mutation_scale=30))
    ax.plot([qx + qw / 2, 6.6], [1.75, 1.75], color=INK, lw=4, zorder=1)
    ax.plot([6.6, 6.6], [ys[-1], ys[0]], color=INK, lw=4, zorder=1)
    fig.savefig(out / "03_found_but_wrong_label.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def d4(out: Path):
    fig, ax = canvas("Missed objects and extra boxes")
    qw, qh, ow, oh = 4.5, 1.55, 4.5, 1.3
    # ---- top band: a REAL object that no box matched
    ax.text(0.4, 7.62, "A REAL object that no box matched:", ha="left", va="center", fontsize=25 * SCALE, fontweight="bold", color=INK)
    box(ax, 2.7, 6.35, qw, qh, "Right-class box,\nscore under 0.3?", GREY, fs=24)
    box(ax, 8.4, 6.35, qw, qh, "Right-class box,\nbarely overlaps?", GREY, fs=24)
    box(ax, 13.6, 6.35, 4.3, oh, "NOTHING FOUND", RED, fs=24)
    arrow(ax, (2.7 + qw / 2, 6.35), (8.4 - qw / 2, 6.35), "NO", RED, label_offset=(0, 0.35))
    arrow(ax, (8.4 + qw / 2, 6.35), (13.6 - 2.15, 6.35), "NO", RED, label_offset=(0, 0.35))
    box(ax, 2.7, 4.2, ow, oh, "MISSED:\nLOW SCORE", RED, fs=25)
    box(ax, 8.4, 4.2, ow, oh, "POOR\nLOCATION", RED, fs=25)
    arrow(ax, (2.7, 6.35 - qh / 2), (2.7, 4.2 + oh / 2), "YES", GREEN, label_offset=(0.62, 0))
    arrow(ax, (8.4, 6.35 - qh / 2), (8.4, 4.2 + oh / 2), "YES", GREEN, label_offset=(0.62, 0))
    ax.plot([0.4, 15.6], [3.1, 3.1], color="#ced4da", lw=3)
    # ---- bottom band: a box with NO real object under it
    ax.text(0.4, 2.7, "A box with NO real object under it:", ha="left", va="center", fontsize=25 * SCALE, fontweight="bold", color=INK)
    box(ax, 8.0, 1.2, 4.7, qh, "Overlaps an object\nalready found?", GREY, fs=24)
    box(ax, 2.3, 1.2, 4.0, oh + 0.1, "EXTRA BOX:\nDUPLICATE", RED, fs=25)
    box(ax, 13.7, 1.2, 4.0, oh + 0.1, "EXTRA BOX:\nBACKGROUND", RED, fs=25)
    arrow(ax, (8.0 - 2.35, 1.2), (2.3 + 2.0, 1.2), "YES", GREEN, label_offset=(0, 0.4))
    arrow(ax, (8.0 + 2.35, 1.2), (13.7 - 2.0, 1.2), "NO", RED, label_offset=(0, 0.4))
    fig.savefig(out / "04_missed_and_extra_boxes.png", dpi=110)
    plt.close(fig)


def main():
    global SCALE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply every font size (default 1.0)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    SCALE = a.scale
    for f in (d1, d2, d3, d4):
        f(a.out_dir)
    print(f"[info] wrote diagrams to {a.out_dir}")


if __name__ == "__main__":
    main()
