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

  **GPU notes:** on a 6GB card like a GTX 1660 Ti, `yolov8n.pt` (default)
  or `yolov8s.pt` both fit comfortably at `--imgsz 640 --batch 16`.
  `yolov8m.pt` can also fit on 6GB but usually needs a smaller batch (try
  `--batch 8`) to avoid an out-of-memory error, and with only ~250 total
  images in this dataset a model that size is likely to overfit anyway —
  `yolov8s.pt` is a reasonable ceiling here. Make sure a CUDA-enabled
  build of PyTorch is installed (`python -c "import torch;
  print(torch.cuda.is_available())"` should print `True`) — if it prints
  `False`, reinstall PyTorch using the command for your CUDA version from
  https://pytorch.org/get-started/locally/ before installing `ultralytics`.
  Pass `--device 0` explicitly to make sure it uses the GPU rather than
  auto-selecting CPU.

  **Training feels too slow (e.g. an hour+ for a small dataset)?**
  `yolov8n.pt` is already the smallest stock model — there's nothing
  smaller in the family to switch to. If it's still slow, the cause is
  almost always one of these, roughly in order of how much it usually
  matters:
  1. **It's silently running on CPU.** By far the most common cause —
     CPU training is typically 10-50x slower than GPU even for the
     smallest model. `train_yolo.py` now checks this and prints a warning
     if `torch.cuda.is_available()` is `False`. Fix as described above,
     then pass `--device 0` explicitly so it can't fall back to CPU.
  2. **No image caching**, so every epoch re-reads and re-decodes every
     image from disk. Add `--cache ram` (fastest; needs enough RAM to
     hold the dataset — this dataset is tiny, so this is basically free)
     or `--cache disk` if RAM is tight.
  3. **Too many dataloader workers on Windows** — the default
     `--workers 8` spawns multiprocessing workers, which has more
     overhead on Windows than Linux and can slow down small datasets.
     Try `--workers 0` or `--workers 4`.

  ```bash
  python train_yolo.py --dataset-dir ../../dataset --device 0 --cache ram --workers 4
  ```

- **`evaluate.py`** — standalone evaluation for any trained checkpoint.
  Runs ultralytics' standard detection metrics (precision, recall, mAP50,
  mAP50-95) overall and per class. Since every class here is really an
  (object, material, color) combination, it also breaks results down two
  other ways and writes all three to CSV:
  - **Per-object** — every material/color variant of the same object
    combined (e.g. all 3 `ballpoint pen` classes into one row), so you can
    see "how good is detection of this object" independent of which
    material variant it is.
  - **Per-material** — every object sharing a material combined (e.g.
    every `plastic` class across all objects into one row), so you can
    see "how good is material recognition" independent of which object
    it's on.
  - Both are **macro-averaged** (every object/material counts equally,
    regardless of how many images it has), so a material used by only one
    object isn't drowned out by a material used by ten.
  - A **weighted composite score** combines the two into a single number:
    `object_weight * object_macro_mAP50-95 + material_weight * material_macro_mAP50-95`,
    tunable with `--object-weight`/`--material-weight` (default 0.5/0.5
    each). This is a custom summary on top of the standard metrics, not a
    replacement for them — the standard overall precision/recall/mAP is
    always reported too.
  - **Per-category** (opt-in via `--category-map`) — gives credit for
    getting the general category right even if the model mixes up the
    exact object within it (e.g. detecting a gel pen as a ballpoint pen
    still correctly found "a pen"). This is a genuinely different kind of
    metric from the others: it's built from the confusion matrix (what got
    predicted as what), not from mAP, since mAP scores each class in
    isolation and structurally can't express "close enough, right
    category." You define the groupings yourself in a small YAML file —
    see `dataset/object_categories.yaml` for the format and a filled-in
    example using this dataset's actual objects (e.g. grouping
    `ballpoint pen` and, once you add it, `gel pen` under `pen`). If that
    file exists, it's used automatically; point `--category-map` at a
    different file to override. Reports precision/recall/F1 (not mAP) per
    category, plus TP/FP/FN counts, and writes `evaluation_per_category.csv`.

  ```bash
  # evaluate against the validation split (default)
  python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../../dataset

  # evaluate against a held-out test split instead
  python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../../dataset --split test

  # weight material recognition more heavily in the composite score
  python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../../dataset --material-weight 0.7 --object-weight 0.3

  # also report category-level accuracy (auto-used if dataset/object_categories.yaml exists)
  python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../../dataset --category-map ../../dataset/object_categories.yaml
  ```

  Writes to `runs/val/<name>/`: `evaluation_report.txt` (overall metrics,
  composite score, per-class/object/material/category tables,
  worst-performing classes and materials called out, and a final summary
  table with every metric at every aggregation level), plus
  `evaluation_per_class.csv`, `evaluation_per_object.csv`,
  `evaluation_per_material.csv`, `evaluation_per_category.csv` (when a
  category map is used), and `evaluation_summary.csv`.

  The bottom of the report always has a complete summary table like:

  ```
  SUMMARY -- COMPLETE RESULTS FOR EVERY METRIC
                                          precision   recall    mAP50  mAP50-95
  overall (instance-weighted)               0.8421   0.7900   0.8650    0.6120
  macro avg across classes                  0.8103   0.7650   0.8390    0.5940
  macro avg across objects                  0.8250   0.7800   0.8500    0.6050
  macro avg across materials                0.7980   0.7500   0.8200    0.5810
  weighted composite (object+material)      0.8115   0.7650   0.8350    0.5930
  ```

- **`resplit_dataset.py`** — rebuilds train/valid/test from scratch at a
  ratio you choose. Three modes:
  - `--mode stratified` (recommended for a normal train/val split of a
    fixed object set): splits each object's own images at the target
    ratio individually, so every object with enough images ends up in
    **every** split. Use this for the usual "train on these objects, then
    check accuracy on held-out photos of them" setup.
  - `--mode object-disjoint` (default): every object (every `Pdfname` id)
    ends up entirely in one split — never split across train and
    valid/test, even if two objects co-occur in the same photo (grouped
    together too). Use this instead when you specifically want to test
    generalization to objects the model has never seen — it trades off
    not being able to validate on some objects at all.
  - `--mode random`: classic per-image random shuffle, ignoring object
    identity, for comparison. Unlike stratified, it doesn't guarantee
    every object appears in every split — with only a few images for some
    objects, a plain random cut can (and did, in testing) leave some of
    them with zero validation images purely by chance.

  Non-destructive by default — writes to `dataset/resplit_preview/`
  (or `--output-dir`) without touching your current train/valid/test.
  Pass `--apply-in-place` to actually replace them (your existing
  train/valid/test are renamed to `<split>.bak_<timestamp>` first, and the
  new split is fully staged before anything existing is touched or moved,
  so a mid-run failure can't leave you with data half-written or lost).

  ```bash
  # recommended: 80/20 split where every object appears in both train and valid
  python resplit_dataset.py --dataset-dir ../../dataset --mode stratified --train-ratio 0.8 --val-ratio 0.2 --apply-in-place

  # preview a 70/20/10 split with no object leakage across splits
  python resplit_dataset.py --dataset-dir ../../dataset --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
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

## Adding a brand new object

1. Photograph the new object (30-40 images, varied rotation/placement/lighting is the target) and get bounding-box labels for them in the usual YOLO `.txt` format.
2. Run the object through the same Gemini labeling process as the others (correcting it in the conversation if the first guess is wrong -- the bottom-up PDF parsing always uses the most recent answer, so an early misfire like guessing "ballpoint pen" before you correct it to "gel pen" is automatically ignored) and save the transcript as a new PDF under `dataset/pdf_for_labels/<new_id>.pdf`.
3. Run `build_dataset.py`. A PDF with no matching class yet is flagged `[ORPHAN PDF]`, and the report now tells you exactly what to do about it -- it parses the PDF's current answer and prints ready-to-paste `data.yaml` name line(s):
   ```
   [ORPHAN PDF] newcam001.pdf exists but no class in data.yaml references it yet.
       It parses as object='gel pen' with 1 material(s)
       ACTION NEEDED: this PDF alone doesn't add a class -- add these line(s) to data.yaml's
       `names` list, bump `nc` by the same amount, and make sure your bounding-box label
       .txt files use the matching new class index(es) (0-indexed, same order as `names`):
         - 'Object gel pen - Material plastic - Color black - Pdfname newcam001'
   ```
4. Add that line to `data.yaml`'s `names` list, bump `nc` by however many lines you added, and make sure the new images' label `.txt` files use the matching class index (0-indexed position in `names`).
5. Drop the new images/labels into any split folder (`train` is fine) and run `resplit_dataset.py --mode stratified` to redistribute everything, including the new object, across train/val properly.
6. Re-run `build_dataset.py` to confirm the object now shows up as `unchanged` (or a harmless proposed rename) instead of `[ORPHAN PDF]`, then train as usual.

Since matching is keyed on the PDF's unique filename, a new object is never confused with an existing one just because it looks similar (e.g. gel pen vs. ballpoint pen) -- that only matters for how well the *model* tells them apart visually, which more/varied training images for both is what helps with.

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
