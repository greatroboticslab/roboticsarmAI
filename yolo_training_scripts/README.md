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

  # train, then immediately evaluate the best checkpoint (val split, plus test if present)
  python train_yolo.py --dataset-dir ../../dataset --epochs 100 --evaluate
  ```

  Common flags: `--device 0` (GPU index / `cpu` / `mps`), `--batch`,
  `--imgsz`, `--patience` (early stopping), `--project`/`--name` (where
  runs are saved), `--resume`, `--evaluate`.

- **`evaluate.py`** — standalone evaluation for any trained checkpoint.
  Runs ultralytics' standard detection metrics (precision, recall, mAP50,
  mAP50-95) overall and per class, and writes both a readable report and a
  CSV.

  ```bash
  # evaluate against the validation split (default)
  python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../../dataset

  # evaluate against a held-out test split instead
  python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../../dataset --split test

  # evaluate against train too, e.g. to compare and check for overfitting
  python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../../dataset --split train
  ```

  Writes `runs/val/<name>/evaluation_report.txt` (overall + per-class
  table, worst classes called out) and `evaluation_per_class.csv`.

- **`resplit_dataset.py`** — rebuilds train/valid/test from scratch at a
  ratio you choose. Two modes:
  - `--mode object-disjoint` (default): every object (every `Pdfname` id)
    ends up entirely in one split — never split across train and
    valid/test, even if two objects co-occur in the same photo (those are
    grouped together too). This avoids validation/test scores being
    inflated by near-duplicate photos of the *same physical item* the
    model already saw in training, and lets you widen the training set
    (e.g. an 80/20 or 90/10 ratio) when you only have a few images per
    object.
  - `--mode random`: classic per-image random shuffle, ignoring object
    identity, for comparison.

  Non-destructive by default — writes to `dataset/resplit_preview/`
  (or `--output-dir`) without touching your current train/valid/test.
  Pass `--apply-in-place` to actually replace them (your existing
  train/valid/test are renamed to `<split>.bak_<timestamp>` first, and the
  new split is fully staged before anything existing is touched or moved,
  so a mid-run failure can't leave you with data half-written or lost).

  ```bash
  # preview a 70/20/10 split with no object leakage across splits
  python resplit_dataset.py --dataset-dir ../../dataset --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

  # inspect dataset/resplit_preview/, then actually use it:
  python resplit_dataset.py --dataset-dir ../../dataset --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 --apply-in-place
  ```

  Its report calls out the achieved vs. target ratio per split, how many
  distinct objects and label instances ended up in each, and explicitly
  flags (or confirms the absence of) any object appearing in more than one
  split.

- **`test_pipeline.py`** — a smoke test for the whole train → evaluate
  pipeline. Runs a throwaway 1-epoch training job and then `evaluate.py`
  against it, and reports PASS/FAIL on each step. It's not a measure of
  model quality (1 epoch tells you nothing about accuracy) — it's there to
  catch a broken `data.yaml`, a bad image, an invalid class id, or an
  `evaluate.py`/ultralytics-version incompatibility in ~1-2 minutes, before
  you find out about it after a multi-hour real training run. This exact
  script is what caught a real bug during development: this ultralytics
  version nests the actual save directory as `runs/detect/<project>/<name>`
  instead of `<project>/<name>`, which silently broke a naive prediction of
  where `best.pt` would land — `train_yolo.py` and `evaluate.py` now always
  ask ultralytics for the real path instead of guessing it.

  ```bash
  python test_pipeline.py --dataset-dir ../../dataset
  # longer/GPU smoke test:
  python test_pipeline.py --dataset-dir ../../dataset --epochs 3 --imgsz 640 --device 0
  ```

## Typical workflow

```bash
cd scripts/yolo_training
python build_dataset.py --dataset-dir ../../dataset
#  -> review dataset/dataset_build_report.txt

# optional: quick sanity check before committing to a real run
python test_pipeline.py --dataset-dir ../../dataset

# optional: widen the training set with a leakage-free resplit
python resplit_dataset.py --dataset-dir ../../dataset --train-ratio 0.8 --val-ratio 0.2 --apply-in-place

python train_yolo.py --dataset-dir ../../dataset --model yolov8n.pt --epochs 100 --evaluate
#  -> review runs/val/exp_eval_val/evaluation_report.txt
```

## Dependencies

```
pip install pdfplumber pyyaml ultralytics
```
