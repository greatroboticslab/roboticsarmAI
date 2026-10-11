#!/usr/bin/env python3
"""
plot_rfdetr_loss.py

Draws training / validation curves from the metrics.csv that RF-DETR writes
into each run folder (e.g. runs/rfdetr/pretrained/metrics.csv). Works while a
run is still training -- just re-run it to refresh the picture.

Panels:
  1. Loss per epoch: training loss, plus validation loss if the run logged it.
  2. Training loss components (classification / box L1 / GIoU).
  3. Validation accuracy per epoch: mAP50, mAP50-95 and F1.
The epoch with the best validation mAP50-95 is marked.

NOTE on validation loss: RF-DETR only computes it when asked
(compute_val_loss=True). train_rfdetr.py now asks for it, so NEW runs have a
val-loss curve. Runs started before that update don't have one -- for those the
plot says so and the validation mAP/F1 curves (which are always logged) are the
thing to look at.

Give several run folders to overlay them (e.g. pretrained vs scratch):

Usage:
    python plot_rfdetr_loss.py runs/rfdetr/pretrained
    python plot_rfdetr_loss.py runs/rfdetr/pretrained runs/rfdetr/scratch      # overlay + compare
    python plot_rfdetr_loss.py runs/rfdetr/pretrained --show                   # also open a window
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "common"))  # shared helpers live in ../common

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

COMPONENTS = [("train/loss_ce", "classification (CE)"), ("train/loss_bbox", "box L1"), ("train/loss_giou", "box GIoU")]


def load_metrics(csv_path: Path) -> dict[str, dict[int, float]]:
    """Return {column: {epoch: mean value}}, averaging the several rows (steps) logged within one epoch."""
    sums: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ep = int(float(row["epoch"]))
            except (KeyError, ValueError):
                continue
            for k, v in row.items():
                if k in ("epoch", "step") or v in ("", None):
                    continue
                try:
                    sums[k][ep].append(float(v))
                except ValueError:
                    pass
    return {k: {ep: sum(v) / len(v) for ep, v in d.items()} for k, d in sums.items()}


def series(m: dict, col: str):
    d = m.get(col)
    if not d:
        return [], []
    xs = sorted(d)
    return [x + 1 for x in xs], [d[x] for x in xs]   # show epochs as 1-based


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="Run folder(s) containing metrics.csv (or the csv itself)")
    ap.add_argument("--out", type=Path, default=None, help="Output image (default: <run>/loss_curves.png, or ./loss_comparison.png for several runs)")
    ap.add_argument("--show", action="store_true", help="Also open an interactive window")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for r in args.runs:
        p = r / "metrics.csv" if r.is_dir() else r
        if not p.exists():
            raise SystemExit(f"No metrics.csv found at {p}")
        runs.append((r.name if r.is_dir() else r.parent.name, p, load_metrics(p)))

    multi = len(runs) > 1
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    ax_loss, ax_comp, ax_acc = axes
    missing_val_loss = []

    for i, (name, _, m) in enumerate(runs):
        c = colors[i % len(colors)]
        pre = f"{name}: " if multi else ""

        x, y = series(m, "train/loss")
        ax_loss.plot(x, y, color=c, lw=2, label=f"{pre}train")
        x, y = series(m, "val/loss")
        if x:
            ax_loss.plot(x, y, color=c, lw=2, ls="--", label=f"{pre}validation")
        else:
            missing_val_loss.append(name)

        for j, (col, label) in enumerate(COMPONENTS):
            x, y = series(m, col)
            if x:
                ax_comp.plot(x, y, color=c, lw=1.8, ls=["-", "--", ":"][j], label=f"{pre}{label}")

        for col, label, ls in (("val/mAP_50_95", "mAP50-95", "-"), ("val/mAP_50", "mAP50", "--"), ("val/F1", "F1", ":")):
            x, y = series(m, col)
            if x:
                ax_acc.plot(x, y, color=c, lw=2, ls=ls, label=f"{pre}{label}")
        x, y = series(m, "val/mAP_50_95")
        if x:
            b = max(range(len(y)), key=lambda k: y[k])
            ax_acc.scatter([x[b]], [y[b]], color=c, s=90, zorder=5, edgecolor="black")
            ax_acc.annotate(f"best {y[b]:.3f}\nepoch {x[b]}", (x[b], y[b]), textcoords="offset points",
                            xytext=(8, -26), fontsize=9)
            print(f"{name}: best val mAP50-95 {y[b]:.4f} at epoch {x[b]} (of {len(x)} epochs logged)")

    ax_loss.set_title("Loss per epoch")
    ax_loss.set_ylabel("loss")
    ax_comp.set_title("Training loss components")
    ax_comp.set_yscale("log")
    ax_acc.set_title("Validation accuracy per epoch")
    ax_acc.set_ylim(bottom=0)
    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    if missing_val_loss:
        ax_loss.text(0.5, 0.97, "no validation loss logged for: " + ", ".join(missing_val_loss)
                     + "\n(runs started before compute_val_loss was enabled)",
                     transform=ax_loss.transAxes, ha="center", va="top", fontsize=8, color="firebrick")
    fig.suptitle("RF-DETR training curves" + (" -- comparison" if multi else f" -- {runs[0][0]}"))
    fig.tight_layout()

    out = args.out or (Path("loss_comparison.png") if multi else runs[0][1].parent / "loss_curves.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"[info] saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"{e}\nInstall matplotlib with:  pip install matplotlib", file=sys.stderr)
        sys.exit(1)
