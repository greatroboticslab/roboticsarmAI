# YOLO training pipeline

Trains a YOLO detector on `dataset/`, using `dataset/pdf_for_labels/*.pdf`
as the source of truth for what each class is actually called.

## `dataset/data.yaml` is the source of truth

People hand-edit `data.yaml` in ways that legitimately diverge from a literal
reading of the PDFs — e.g. for the claw hammer, only `metal` was kept for
the head/neck and only `rubber` for the grip, even though a part-by-part PDF
answer might list the head and neck as two separate `metal` entries. That's
an intentional curation choice, not a mistake.

So `build_dataset.py` is **read-only by default**: it never edits
`data.yaml`, and `data.corrected.yaml` starts as an exact copy. It only ever
*reports* what each PDF's current (most-recent) answer says, so you can
judge for yourself whether a difference is a real missed update or just how
you chose to label that object. Pass `--apply-renames` to opt into writing
the safe subset of changes (see below).

## Why the PDF step exists

Each PDF in `dataset/pdf_for_labels/` is a Gemini labeling transcript for one
object. It's not always a single answer — sometimes it contains one or more
human corrections appended below the original guess, e.g.:

```
Response: Object: wire stripper | Material: plastic, metal | Color: red
...
User prompt: thats a pen ands its plastic rubber near the tip and transparent plastic
Response: Object: ballpoint pen | Material: plastic, rubber | Color: red, clear
```

The **bottom-most** response in the PDF is always the current, correct
answer — anything above it has been superseded. `pdf_label_parser.py` walks
each PDF starting from the bottom "Response:" turn and works upward, only
skipping to an earlier turn if the bottom one can't be parsed at all (e.g. a
PDF text-extraction artifact truncated it).

## Scripts

- **`pdf_label_parser.py`** — library only. Given one PDF, returns the
  object name + material/color list from its most recent answer. Run
  directly for debugging: `python pdf_label_parser.py path/to/some.pdf`

- **`build_dataset.py`** — run this first. For every PDF, it:
  1. Extracts the current (most recent) label.
  2. Matches it to the existing classes in `dataset/data.yaml` via the
     `Pdfname <id>` suffix baked into each class name.
  3. If the number of materials matches what's already in `data.yaml`,
     it's a **[PROPOSED RENAME]** — reported, but only written to
     `data.corrected.yaml` if you pass `--apply-renames`. Class *indices*
     are never changed either way, so existing YOLO label `.txt` files
     always stay valid. Matching an old entry to the right PDF material
     uses a fallback chain so a manual edit to material *or* color doesn't
     get mismatched:
       1. material + color both match → confident match, no warning
       2. material matches, color differs → matched by material, flagged
          as a possible manual color correction to double-check
       3. color matches, material differs → matched by color, flagged as
          a possible manual material correction to double-check
       4. neither matches → last-resort positional match, flagged low confidence
  4. If the number of materials in the PDF's current answer doesn't match
     what's in `data.yaml`, that's reported as **[COUNT DIFFERS FROM PDF]**
     — purely informational, never applied automatically regardless of
     `--apply-renames`. This is common and often intentional (e.g. only
     labeling the material of the part actually visible, or merging parts
     that share a material) — use your judgement on whether it needs a
     manual fix.
  5. Writes `dataset/data.corrected.yaml` and
     `dataset/dataset_build_report.txt` (full detail of every entry).
  6. Audits every split's `images/` vs `labels/` folders: orphan images,
     orphan labels, out-of-range class ids, malformed label lines.

  ```bash
  # dry run (default): reports everything, data.corrected.yaml == data.yaml
  python build_dataset.py --dataset-dir ../../dataset

  # opt in to writing the safe, count-matched renames
  python build_dataset.py --dataset-dir ../../dataset --apply-renames

  # exit non-zero if anything still differs from data.yaml, e.g. for CI:
  python build_dataset.py --dataset-dir ../../dataset --strict
  ```

  Read `dataset/dataset_build_report.txt` after every run before training —
  it never overwrites your manual curation unless you ask it to.

- **`train_yolo.py`** — trains with ultralytics YOLO. Prefers
  `dataset/data.corrected.yaml` if present (falls back to the raw
  `data.yaml` with a warning if `build_dataset.py` hasn't been run yet).

  ```bash
  pip install ultralytics
  python train_yolo.py --dataset-dir ../../dataset --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16
  ```

  Common flags: `--device 0` (GPU index / `cpu` / `mps`), `--batch`,
  `--imgsz`, `--patience` (early stopping), `--project`/`--name` (where
  runs are saved), `--resume`.

## Typical workflow

```bash
cd scripts/yolo_training
python build_dataset.py --dataset-dir ../../dataset
#  -> review dataset/dataset_build_report.txt, resolve any [CONFLICT] entries
#     by re-checking/re-drawing bounding boxes for that object, if needed
python train_yolo.py --dataset-dir ../../dataset --model yolov8n.pt --epochs 100
```

## Dependencies

```
pip install pdfplumber pyyaml ultralytics
```
