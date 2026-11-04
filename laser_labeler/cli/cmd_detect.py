"""`detect`: run laser detection and measurement with no AI. For tuning on real images."""
from __future__ import annotations

from pathlib import Path

import cv2

from cli.common import data_dir_for, load_cli_settings
from core.camera import image_stats, load_image
from core.tools.laser_detect import detect_laser
from core.tools.measure_laser import measure_laser
from core.tools.overlay import draw_boxes


def add_parser(subparsers):
    p = subparsers.add_parser("detect", help="find and measure laser spots without any AI")
    p.add_argument("images", nargs="+")
    p.add_argument("--out", help="folder for overlay pictures (default: <data>/detect)")
    p.add_argument("--settings")
    p.set_defaults(func=run)


def run(args) -> int:
    settings = load_cli_settings(args.settings)
    out = Path(args.out) if args.out else data_dir_for(settings) / "detect"
    out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = load_image(path)
        stats = image_stats(image.rgb)
        det = detect_laser(image.rgb, stats, max_candidates=settings["run"]["max_candidates"])
        kind = "grayscale" if image.is_grayscale else "color"
        print(f"{path}: {kind}, background {stats.background_level:.3f}, noise {stats.noise_level:.4f}, "
              f"{len(det.candidates)} spot(s), {len(det.lines)} beam line(s)")
        for c in det.candidates:
            m = measure_laser(image.rgb, c.box, stats, image.is_grayscale)
            print(f"  {c.id}: box {c.box} peak {c.peak:.2f} area {c.area}px scatter {m['scatter_ratio']:.2f} "
                  f"halo {m['halo_radius_px']:.0f}px clipped {m['clipped_fraction']:.0%} conf {m['confidence']:.2f}")
        overlay = draw_boxes(image.rgb, [(c.id, c.box) for c in det.candidates])
        target = out / f"{Path(path).stem}_detect.png"
        cv2.imwrite(str(target), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  overlay: {target}")
    return 0
