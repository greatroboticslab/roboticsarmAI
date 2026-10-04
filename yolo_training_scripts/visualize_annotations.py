#!/usr/bin/env python3
"""
visualize_annotations.py

Renders a clean, presentation-ready version of a split's images with their
ground-truth bounding boxes drawn on -- distinct colors per class, readable
labels (using the short "object (material, color)" form, not the full raw
"Object X - Material Y - Color Z - Pdfname W" class string), rounded boxes
with a light drop border for contrast against any background. Meant for
reports/slides/sharing with someone who doesn't need to parse YOLO .txt
files, not for debugging exact box pixel placement.

Can also build a single contact-sheet grid image combining several samples,
and/or a simple self-contained HTML gallery page to browse the results.

Usage:
    # annotate every image in the validation split
    python visualize_annotations.py --dataset-dir ../dataset --split valid

    # just a representative sample of 12, plus a contact sheet and a gallery page
    python visualize_annotations.py --dataset-dir ../dataset --split valid --limit 12 --grid --html-gallery

    # test split, custom output location
    python visualize_annotations.py --dataset-dir ../dataset --split test --output-dir ../dataset/preview_test
"""

from __future__ import annotations

import argparse
import colorsys
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from yolo_common import (
    resolve_data_yaml, load_data_yaml, find_split_dirs, list_image_label_pairs,
    read_label_class_ids, parse_class_name,
)

FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\calibrib.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def load_font(size: int, explicit_path: Path | None = None) -> ImageFont.FreeTypeFont:
    candidates = [str(explicit_path)] + FONT_CANDIDATES if explicit_path else FONT_CANDIDATES
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    print("[warn] no TrueType font found on this system -- falling back to a basic bitmap font "
          "(labels will look blocky). Pass --font /path/to/font.ttf to fix this.")
    return ImageFont.load_default()


def build_color_map(class_names: list[str]) -> dict[int, tuple[int, int, int]]:
    """Assign each class a visually distinct, deliberately not-neon color, deterministically
    (same class always gets the same color across runs), by spacing hues evenly around the wheel."""
    n = max(len(class_names), 1)
    colors = {}
    for i in range(len(class_names)):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)  # moderate saturation/value: readable, not garish
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def short_label(class_name: str) -> str:
    """'Object ballpoint pen - Material plastic - Color red - Pdfname Tg2w...' -> 'ballpoint pen (plastic, red)'"""
    parts = parse_class_name(class_name)
    obj = parts["object"] or class_name
    descriptors = [d for d in (parts["material"], parts["color"]) if d]
    return f"{obj} ({', '.join(descriptors)})" if descriptors else obj


def draw_single_box(
    draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], color: tuple[int, int, int],
    label: str, font: ImageFont.FreeTypeFont, box_width: int = 3,
):
    """Draw one rounded box with a white contrast border and a padded label pill above it
    (or just inside the top edge if the box touches the top of the image). Shared by ground-truth
    and prediction rendering so both look identical apart from color/label content."""
    x1, y1, x2, y2 = box
    draw.rounded_rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], radius=6, outline=(255, 255, 255, 230), width=box_width + 2)
    draw.rounded_rectangle([x1, y1, x2, y2], radius=6, outline=color + (255,), width=box_width)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    pad = 4
    label_y2 = y1
    label_y1 = label_y2 - text_h - 2 * pad
    if label_y1 < 0:  # box touches the top edge -- put the label inside/below instead
        label_y1 = y1
        label_y2 = label_y1 + text_h + 2 * pad
    draw.rounded_rectangle([x1, label_y1, x1 + text_w + 2 * pad, label_y2], radius=4, fill=color + (235,))
    text_color = (255, 255, 255) if sum(color) < 400 else (20, 20, 20)
    draw.text((x1 + pad, label_y1 + pad - text_bbox[1]), label, font=font, fill=text_color)


def draw_annotated_image(
    img_path: Path, label_path: Path, class_names: list[str], color_map: dict, font: ImageFont.FreeTypeFont,
    box_width: int = 3,
) -> tuple[Image.Image, list[str]]:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    labels_drawn = []

    if not label_path.exists():
        return img, labels_drawn

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label = short_label(cls_name)
        labels_drawn.append(label)
        color = color_map.get(cls_id, (220, 40, 40))

        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
        draw_single_box(draw, (x1, y1, x2, y2), color, label, font, box_width)

    return img, labels_drawn


def build_contact_sheet(annotated: list[tuple[Image.Image, str]], cols: int = 4, thumb_size: int = 320) -> Image.Image:
    pad = 12
    caption_h = 24
    rows = (len(annotated) + cols - 1) // cols
    sheet_w = cols * (thumb_size + pad) + pad
    sheet_h = rows * (thumb_size + caption_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    font = load_font(13)

    for i, (img, caption) in enumerate(annotated):
        col, row = i % cols, i // cols
        thumb = img.copy()
        thumb.thumbnail((thumb_size, thumb_size))
        x = pad + col * (thumb_size + pad)
        y = pad + row * (thumb_size + caption_h + pad)
        offset_x = x + (thumb_size - thumb.width) // 2
        sheet.paste(thumb, (offset_x, y))
        draw.text((x, y + thumb_size + 4), caption[:40], font=font, fill=(40, 40, 40))

    return sheet


def build_html_gallery(output_dir: Path, entries: list[tuple[str, str]]) -> Path:
    """entries: [(image_filename, caption), ...]"""
    cards = "\n".join(
        f'<div class="card"><img src="{fname}" loading="lazy"><div class="cap">{caption}</div></div>'
        for fname, caption in entries
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Annotated samples</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; background: #fafafa; margin: 24px; }}
  h1 {{ font-size: 18px; color: #222; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
  .card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.12); }}
  .card img {{ width: 100%; display: block; }}
  .cap {{ padding: 8px 10px; font-size: 13px; color: #333; }}
</style></head>
<body>
  <h1>Annotated samples ({len(entries)})</h1>
  <div class="grid">
    {cards}
  </div>
</body></html>
"""
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml, overrides --dataset-dir auto-detection")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: <dataset-dir>/annotated_preview/<split>")
    ap.add_argument("--limit", type=int, default=None, help="Only render a random sample of N images instead of the whole split")
    ap.add_argument("--seed", type=int, default=42, help="Random seed used when --limit is set, for a reproducible sample")
    ap.add_argument("--box-width", type=int, default=3)
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--font", type=Path, default=None, help="Path to a .ttf font to use instead of auto-detecting one")
    ap.add_argument("--grid", action="store_true", help="Also build a single contact-sheet image combining the rendered samples")
    ap.add_argument("--grid-cols", type=int, default=4)
    ap.add_argument("--html-gallery", action="store_true", help="Also build a self-contained index.html gallery page")
    args = ap.parse_args()

    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)
    class_names = load_data_yaml(data_yaml)["names"]
    color_map = build_color_map(class_names)
    font = load_font(args.font_size, args.font)

    split_dirs = find_split_dirs(args.dataset_dir)
    if args.split not in split_dirs:
        raise SystemExit(f"Split '{args.split}' not found under {args.dataset_dir} (found: {sorted(split_dirs)})")

    pairs = list_image_label_pairs(split_dirs[args.split])
    if args.limit is not None and args.limit < len(pairs):
        rng = random.Random(args.seed)
        pairs = rng.sample(pairs, args.limit)
        pairs.sort(key=lambda p: p[0].name)

    output_dir = args.output_dir or (args.dataset_dir / "annotated_preview" / args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(pairs)} image(s) from '{args.split}' -> {output_dir}")
    grid_entries: list[tuple[Image.Image, str]] = []
    html_entries: list[tuple[str, str]] = []
    n_no_objects = 0

    for img_path, label_path in pairs:
        annotated, labels = draw_annotated_image(img_path, label_path, class_names, color_map, font, args.box_width)
        out_path = output_dir / img_path.name
        annotated.save(out_path, quality=92)

        caption = ", ".join(sorted(set(labels))) if labels else "(no objects labeled -- background image)"
        if not labels:
            n_no_objects += 1
        if args.grid:
            grid_entries.append((annotated, caption))
        if args.html_gallery:
            html_entries.append((img_path.name, caption))

    print(f"Wrote {len(pairs)} annotated image(s) ({n_no_objects} with no labeled objects).")

    if args.grid and grid_entries:
        sheet = build_contact_sheet(grid_entries, cols=args.grid_cols)
        grid_path = output_dir / "_contact_sheet.jpg"
        sheet.save(grid_path, quality=90)
        print(f"Contact sheet: {grid_path}")

    if args.html_gallery and html_entries:
        gallery_path = build_html_gallery(output_dir, html_entries)
        print(f"HTML gallery: {gallery_path}")


if __name__ == "__main__":
    main()
