# yolo/ -- YOLO (ultralytics) training, evaluation and error sorting

Trains a YOLO detector on `dataset/`, evaluates it, and sorts every validation
image into *correct* or into the kind of mistake it made. Dataset preparation
(PDF labels, resplitting) lives in [`../common/`](../common/README.md); the
transformer model is in [`../transformer/`](../transformer/README.md).

```
pip install ultralytics
python train_yolo.py --dataset-dir ../../dataset --model yolov8s.pt --epochs 100 --device 0 --cache ram --evaluate
python sort_yolo_errors.py --dataset-dir ../../dataset --weights runs/detect/runs/train/exp/weights/best.pt
```

## How the evaluation works

![How one image is evaluated](../diagrams/01_evaluation_pipeline.png)

![One class name, scored four ways](../diagrams/02_label_levels.png)

- `evaluate.py` scores with ultralytics' **mAP** (and precision/recall), then
  re-pools the classes by object / material / color and combines them into the
  weighted composite.
- `sort_yolo_errors.py` scores with the **shared sorting rules** at one
  confidence threshold, which is what makes it comparable with RF-DETR.

![A box was found on a real object](../diagrams/03_found_but_wrong_label.png)

![Missed objects and extra boxes](../diagrams/04_missed_and_extra_boxes.png)

## Scripts

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

  **Augmentation controls.** Ultralytics augments images on the fly during
  training. By default that includes color jitter (hue/saturation/brightness),
  random left-right flips, and mosaic stitching. If color, brightness or laser
  glow/diffraction patterns are part of what distinguishes your classes,
  turn the color part off and lean on geometric augmentation instead:

  ```bash
  python train_yolo.py --dataset-dir ../../dataset --no-color-aug --degrees 10 --translate 0.15 --fliplr 0
  ```

  Other flags: `--hsv-h/--hsv-s/--hsv-v`, `--degrees`, `--translate`, `--scale`,
  `--shear`, `--fliplr`, `--flipud`, `--mosaic`. Unset flags keep ultralytics'
  defaults. Run the default and the modified version under different `--name`
  values and compare their `evaluation_report.txt` to see which actually helps.

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
  - **Per-color** — every object/material sharing a color combined.
  - All three are **macro-averaged** (every object/material/color counts equally,
    regardless of how many images it has), so a material used by only one
    object isn't drowned out by a material used by ten.
  - A **weighted composite score** combines the three into a single number,
    prioritised object > material > color:
    `object_weight * object_macro_mAP50-95 + material_weight * material_macro_mAP50-95 + color_weight * color_macro_mAP50-95`,
    tunable with `--object-weight`/`--material-weight`/`--color-weight`
    (defaults 0.6 / 0.3 / 0.1, normalized to sum to 1; color is deliberately
    the least-weighted term). This is a custom summary on top of the standard metrics, not a
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

- **`sort_yolo_errors.py`** -- runs the trained model over a split and **sorts
  every image into folders**: `correct/` and `failure/<type>/` (wrong object,
  wrong material, wrong color, wrong material and color, same-category object,
  missed, low score, poor localization, false positive background / duplicate).
  Each image is drawn with the true boxes (green) and the model's (red), and the
  run also writes `image_summary.csv`, `errors_report.csv`, `confusions.csv`,
  `summary.txt` and `summary.json`. Finer-grained successor to
  `find_prediction_errors.py` below (which is kept as it was).

  ```bash
  python sort_yolo_errors.py --dataset-dir ../../dataset --weights runs/detect/runs/train/exp/weights/best.pt
  python sort_yolo_errors.py --dataset-dir ../../dataset --weights .../best.pt --split test --conf 0.2
  ```
  Output goes to `<run folder>/error_sort_<split>/`. Use `--imgsz` to match your
  training size, `--plain` to copy the original images without boxes drawn on.

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

- **`compare_predictions.py`** — picks ONE representative image per
  distinct object and renders it twice, into two separate folders, so the
  human labels and the model's actual behavior are directly comparable
  side by side:
  - `ground_truth/` — the human-labeled ("control") boxes
  - `model_predictions/` — what your trained model actually detects on
    that same image (needs `--weights`; this runs real inference, it's not
    just a re-styling of the ground truth)

  Both folders use matching filenames (named after the object, e.g.
  `rubber_duck.jpg`) so they're easy to open side by side. Rarest objects
  get first pick of an unused image, so a common object's photos don't
  crowd out a rare one's only appearance.

  ```bash
  python compare_predictions.py --weights runs/detect/runs/train/exp/weights/best.pt --dataset-dir ../../dataset

  # if predictions look empty, the model may just be under-confident -- lower the threshold
  python compare_predictions.py --weights .../best.pt --dataset-dir ../../dataset --conf 0.1
  ```

  Writes to `dataset/gt_vs_pred_preview/` (or `--output-dir`). Tested
  end-to-end against the real dataset with a real (if only lightly
  trained) checkpoint — the mechanics (selection, matched filenames,
  rendering) all confirmed working; with a properly trained model the
  `model_predictions/` folder should show real boxes instead of empty
  images.

- **`find_prediction_errors.py`** — runs your model against a whole split,
  matches every prediction to the ground truth by IoU, and exports ONLY
  the images where something went wrong (correct detections are skipped
  entirely). Each exported image shows the true box in **green** and the
  predicted box(es) in **red** overlaid on the same photo, so the true
  value is right there next to what the model actually said. Three error
  types:
  - `MISSED` — a real object with no matching prediction at all
  - `WRONGCLASS` — a prediction overlaps a real object but names the wrong
    class (e.g. predicted "gel pen" where the true label is "ballpoint pen")
  - `FALSEPOS` — a prediction with no matching real object nearby at all

  ```bash
  python find_prediction_errors.py --weights runs/detect/runs/train/exp/weights/best.pt --dataset-dir ../../dataset

  # loosen matching / confidence if you're not sure what threshold is right
  python find_prediction_errors.py --weights .../best.pt --dataset-dir ../../dataset --conf 0.1 --iou-match 0.3
  ```

  Writes annotated images + `errors_report.csv` + `errors_report.txt` to
  `dataset/prediction_errors/` (or `--output-dir`), and automatically zips
  the whole folder to `prediction_errors.zip` right next to it — ready to
  share without an extra step. The IoU-matching logic (the part that
  decides MISSED vs WRONGCLASS vs FALSEPOS) was unit-tested against known
  synthetic boxes covering all three cases, confirming correct behavior;
  end-to-end rendering was verified against the real dataset with a real
  checkpoint.

- **`compare_model_errors.py`** — answers "did this change actually help?"
  by running TWO checkpoints (e.g. a baseline vs. a `--no-color-aug`
  retrain) against the same split and classifying every image into:
  - `fixed_by_b/` — model A got it wrong, model B got it right
  - `regressed_by_b/` — model A got it right, model B got it wrong (worth
    checking even when B looks better overall — a net improvement can
    still hide a new problem)
  - `still_wrong/` — both models got it wrong

  Each image overlays all three: ground truth in **green**, model A's
  prediction in **amber**, model B's prediction in **red**, so you see
  exactly what changed on one photo instead of comparing two folders by eye.

  ```bash
  python compare_model_errors.py \
      --weights-a runs/detect/runs/train/baseline/weights/best.pt \
      --weights-b runs/detect/runs/train/noColorAug/weights/best.pt \
      --label-a baseline --label-b no_color_aug \
      --dataset-dir ../../dataset --split valid
  ```

  Writes to `dataset/model_comparison/` (or `--output-dir`) plus
  `comparison_report.csv`/`.txt` with a net fixed-vs-regressed count, and
  zips the whole thing automatically. Tested end-to-end against the real
  dataset with two real (if only lightly trained) checkpoints — one
  baseline, one with `--no-color-aug --degrees 10 --translate 0.15
  --fliplr 0` — confirming the classification and three-color rendering
  both work correctly.

- **`run_full_error_report.py`** — the one-command version of the above:
  runs `find_prediction_errors.py` for BOTH checkpoints and
  `compare_model_errors.py` for the two of them together, across BOTH the
  train and valid splits (and test, if you have one), and zips everything
  into a single file. Each checkpoint is loaded once and reused across
  every split, not reloaded per split. Omit `--weights-b` to just get both
  splits for one model, no comparison.

  ```bash
  python run_full_error_report.py \
      --weights-a runs/detect/runs/train/exp-4/weights/best.pt --label-a baseline \
      --weights-b runs/detect/runs/train/noColorAug/weights/best.pt --label-b no_color_aug \
      --dataset-dir ../../dataset
  ```

  Writes `dataset/full_error_report/` containing
  `baseline_errors/{train,valid}/`, `no_color_aug_errors/{train,valid}/`,
  and `baseline_vs_no_color_aug/{train,valid}/` (each with its own
  report), plus a top-level `SUMMARY.txt`, all zipped to
  `full_error_report.zip`. `find_prediction_errors.py` and
  `compare_model_errors.py` had their core logic factored out into
  reusable functions to support this without duplicating code; both were
  regression-tested standalone afterward to confirm nothing broke, and the
  full orchestrator was verified end-to-end against the real dataset with
  two real checkpoints across both splits.

- **`export_weights.py`** — exports a trained checkpoint to a deployment
  format once you're happy with it (train → evaluate → export). ONNX by
  default, since it's the most portable choice for running inference
  outside Python/ultralytics; also supports TorchScript, TensorRT
  (`engine`), OpenVINO, CoreML, TFLite, and others ultralytics supports.

  ```bash
  # plain ONNX export
  python export_weights.py --weights runs/detect/runs/train/exp/weights/best.pt

  # simplified, dynamic-batch, FP16 ONNX -- smaller/faster, matches your training imgsz
  python export_weights.py --weights .../best.pt --imgsz 640 --dynamic --simplify --quantize 16

  # INT8 needs representative calibration images -- point it at your dataset
  python export_weights.py --weights .../best.pt --quantize 8 --dataset-dir ../../dataset
  ```

  Tested end-to-end: plain ONNX export works, and `--quantize 16` roughly
  halves the file size as expected (11.6 MB → 6.1 MB in testing here).

- **`upload_to_roboflow.py`** — uploads this local dataset (images + YOLO
  labels) to an existing Roboflow project, e.g. to train using Roboflow's
  own hosted training instead of (or in addition to) `train_yolo.py`
  locally. Copies just the standard `train/valid/test` + `data.yaml` pieces
  into a clean staging folder first (leaving out `pdf_for_labels/`, backup
  folders, and reports that would confuse Roboflow's uploader). Requires
  `pip install roboflow` and a Roboflow API key (from your account
  settings) set as the `ROBOFLOW_API_KEY` environment variable or passed
  with `--api-key`.

  Non-destructive by default — previews what would be staged/uploaded
  (image counts, class list) without calling the Roboflow API. Pass
  `--upload` once you've checked the preview.

  ```bash
  # preview only -- stages a clean copy, prints counts/classes, uploads nothing
  python upload_to_roboflow.py --dataset-dir ../../dataset --project your-project-slug

  # try a small batch first (recommended before uploading everything)
  python upload_to_roboflow.py --dataset-dir ../../dataset --project your-project-slug --limit 5 --upload

  # upload everything
  python upload_to_roboflow.py --dataset-dir ../../dataset --project your-project-slug --upload
  ```

  This only adds images+annotations to the project's raw pool. To actually
  train on Roboflow afterward, "Generate" a new Version in the Roboflow web
  UI (pick your split/preprocessing/augmentation settings there) and hit
  Train.

## Results (fill in after your runs)

Weights: `______________`   Dataset split: `valid`   conf `0.3`   IoU `0.5`

| metric | value |
|---|---|
| precision | |
| recall | |
| mAP50 | |
| mAP50-95 | |
| composite (object/material/color) | |
| images fully correct | |
| objects found with exact class | |

![Training curves](results/training_curves.png)
*Fill in by copying ultralytics' `runs/detect/runs/train/exp/results.png` here as `results/training_curves.png`.*

| failure type | images filed here | individual mistakes |
|---|---|---|
| Wrong object | | |
| Wrong object, same category | | |
| Wrong material | | |
| Wrong color | | |
| Wrong material and color | | |
| Wrong class, same attributes | | |
| Missed: nothing found | | |
| Missed: score too low | | |
| Poor localization | | |
| False positive: background | | |
| False positive: duplicate | | |

## Results: example images by error type

Each slot below shows 2 real examples (true boxes **green**, YOLO boxes **red**). They are placeholders until you fill them in:

```bash
# 1. sort the images (writes correct/ and failure/<type>/ folders)
python sort_yolo_errors.py --dataset-dir ../../dataset --weights runs/detect/runs/train/exp/weights/best.pt
# 2. copy 2 examples per type into this README's picture slots
python ../common/collect_examples.py --sorted-dir runs/detect/runs/train/exp/error_sort_valid --out-dir results
```

`results/examples_captions.md` is written alongside the pictures with a one-line caption for each (true class -> what the model said); paste them under the images if you want captions.

### Correct

Every object was found with the exact right class, and there are no extra boxes.

| example 1 | example 2 |
|---|---|
| ![Correct 1](results/correct/example_1.jpg) | ![Correct 2](results/correct/example_2.jpg) |

*What happened:* _fill in after the run_

### Wrong object

A box sits on a real object but names a different object (a pen called a hammer). *Wrong class.*

| example 1 | example 2 |
|---|---|
| ![Wrong object 1](results/wrong_object/example_1.jpg) | ![Wrong object 2](results/wrong_object/example_2.jpg) |

*What happened:* _fill in after the run_

### Wrong object, same category

Wrong object, but both belong to the same category in `object_categories.yaml` (gel pen vs ballpoint pen). A milder kind of wrong object. *Wrong class.*

| example 1 | example 2 |
|---|---|
| ![Wrong object, same category 1](results/wrong_object_same_category/example_1.jpg) | ![Wrong object, same category 2](results/wrong_object_same_category/example_2.jpg) |

*What happened:* _fill in after the run_

### Wrong material

Right object and right color, wrong material (plastic vs rubber). *Wrong class.*

| example 1 | example 2 |
|---|---|
| ![Wrong material 1](results/wrong_material/example_1.jpg) | ![Wrong material 2](results/wrong_material/example_2.jpg) |

*What happened:* _fill in after the run_

### Wrong color

Right object and right material, wrong color. *Wrong class.*

| example 1 | example 2 |
|---|---|
| ![Wrong color 1](results/wrong_color/example_1.jpg) | ![Wrong color 2](results/wrong_color/example_2.jpg) |

*What happened:* _fill in after the run_

### Wrong material and color

Right object, but both material and color are wrong. *Wrong class.*

| example 1 | example 2 |
|---|---|
| ![Wrong material and color 1](results/wrong_material_and_color/example_1.jpg) | ![Wrong material and color 2](results/wrong_material_and_color/example_2.jpg) |

*What happened:* _fill in after the run_

### Wrong class, same attributes

A different class id whose object, material and color are all identical (two `Pdfname` ids that describe the same thing). Usually a labeling-data issue, not a model issue. *Wrong class.*

| example 1 | example 2 |
|---|---|
| ![Wrong class, same attributes 1](results/wrong_class_same_attributes/example_1.jpg) | ![Wrong class, same attributes 2](results/wrong_class_same_attributes/example_2.jpg) |

*What happened:* _fill in after the run_

### Missed: nothing found

A real object with no usable box on it at all. *False negative.*

| example 1 | example 2 |
|---|---|
| ![Missed: nothing found 1](results/missed_no_detection/example_1.jpg) | ![Missed: nothing found 2](results/missed_no_detection/example_2.jpg) |

*What happened:* _fill in after the run_

### Missed: score too low

The model did find it, with the right class in the right place, but its score was under the cut-off (`--conf`, default 0.3). Lowering `--conf` would have caught it. *False negative.*

| example 1 | example 2 |
|---|---|
| ![Missed: score too low 1](results/missed_low_confidence/example_1.jpg) | ![Missed: score too low 2](results/missed_low_confidence/example_2.jpg) |

*What happened:* _fill in after the run_

### Poor localization

Right class, but the box overlaps the real object too little (IoU between 0.1 and 0.5). *False negative.*

| example 1 | example 2 |
|---|---|
| ![Poor localization 1](results/poor_localization/example_1.jpg) | ![Poor localization 2](results/poor_localization/example_2.jpg) |

*What happened:* _fill in after the run_

### False positive: background

A box where there is no real object. *False positive.*

| example 1 | example 2 |
|---|---|
| ![False positive: background 1](results/false_positive_background/example_1.jpg) | ![False positive: background 2](results/false_positive_background/example_2.jpg) |

*What happened:* _fill in after the run_

### False positive: duplicate

A second box on an object that was already found. *False positive.*

| example 1 | example 2 |
|---|---|
| ![False positive: duplicate 1](results/false_positive_duplicate/example_1.jpg) | ![False positive: duplicate 2](results/false_positive_duplicate/example_2.jpg) |

*What happened:* _fill in after the run_

