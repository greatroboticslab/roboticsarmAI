# transformer/ -- RF-DETR (Roboflow vision transformer)

[RF-DETR](https://github.com/roboflow/rf-detr) is a transformer detector built
on a DINOv2 vision-transformer backbone. These scripts train it on the **same**
`dataset/` as the YOLO scripts (same `data.corrected.yaml`, same label files),
evaluate it, plot its loss curves, and sort every validation image into
*correct* or the kind of mistake it made, using the same rules as YOLO.
Dataset preparation lives in [`../common/`](../common/README.md).

```
pip install "rfdetr[train]" matplotlib
python train_rfdetr.py --dataset-dir ../../dataset --model small --epochs 100 --patience 15 --device cuda --no-amp --name pretrained
python sort_rfdetr_errors.py --dataset-dir ../../dataset --weights runs/rfdetr/pretrained
python plot_rfdetr_loss.py runs/rfdetr/pretrained
```

## How the evaluation works

![How one image is evaluated](../diagrams/01_evaluation_pipeline.png)

![One class name, scored four ways](../diagrams/02_label_levels.png)

- `train_rfdetr.py` reports RF-DETR's own **COCO mAP**, plus **precision / recall /
  F1** at a fixed confidence and IoU, broken down per class / object / material /
  color / category, with the weighted composite. (The F1 numbers are not
  directly comparable to YOLO's mAP columns.)
- `sort_rfdetr_errors.py` scores with the **shared sorting rules**, which *are*
  directly comparable with `sort_yolo_errors.py`.

![A box was found on a real object](../diagrams/03_found_but_wrong_label.png)

![Missed objects and extra boxes](../diagrams/04_missed_and_extra_boxes.png)

## Scripts

- **`train_rfdetr.py`** — trains and evaluates Roboflow's **RF-DETR**, a
  vision-transformer (DINOv2 backbone) detector, on the same `dataset/` the
  YOLO scripts use. Same inputs (`data.corrected.yaml` if present, else
  `data.yaml`; same `train/valid/test` YOLO label folders), nothing in
  `dataset/` is modified — it stages a copy under `<run>/dataset_staged`
  because RF-DETR wants a folder with a file literally named `data.yaml`.

  ```bash
  pip install "rfdetr[train]"

  # train + evaluate (val split, plus test if the dataset has one)
  python train_rfdetr.py --dataset-dir ../../dataset --model small --epochs 50 --device cuda

  # GTX 1660 Ti / 16-series: full fp32 avoids fp16 errors; lower --batch if out of memory
  python train_rfdetr.py --dataset-dir ../../dataset --model small --device cuda --no-amp --batch 4

  # evaluate an existing checkpoint only
  python train_rfdetr.py --dataset-dir ../../dataset --eval-only --weights runs/rfdetr/exp/checkpoint_best_total.pth
  ```

  Models: `--model nano|small|medium|base|large` (default `small`; with ~250
  images, `nano`/`small` are the sensible choices). Effective batch size is
  `--batch` x `--grad-accum` (default 4 x 4 = 16). Early stopping via
  `--patience` (default 15, 0 disables). Pretrained COCO weights download
  automatically on first run (needs internet); `--from-scratch` skips them
  but is almost always far worse on a dataset this small.

  Evaluation writes to `runs/rfdetr/<name>/eval_<split>/`:
  - RF-DETR's own COCO metrics (mAP50, mAP50-95, ...) at the top of
    `evaluation_report_<split>.txt`.
  - The same per-class / per-object / per-material / per-color (and
    per-category, via `object_categories.yaml`) breakdown as `evaluate.py`,
    plus the object > material > color weighted composite — **but as
    precision / recall / F1 at `--conf 0.3` and IoU 0.5, not mAP**, because
    they're computed from the model's actual predictions. Don't compare these
    F1 numbers directly with `evaluate.py`'s mAP columns; compare RF-DETR
    runs to each other (change `--conf` to see the precision/recall trade-off).
  - CSVs: `evaluation_per_{class,object,material,color,category}.csv`,
    `evaluation_summary.csv`.

  **Training notes.** With ~1250 images, `--model small` takes roughly 6 min per
  epoch on a GTX 1660 Ti; `--model nano` is faster. To test whether pretrained
  weights matter, run the same command twice with different `--name`:

  ```bash
  python train_rfdetr.py --dataset-dir ../../dataset --model small --epochs 100 --patience 15 --device cuda --no-amp --name pretrained
  python train_rfdetr.py --dataset-dir ../../dataset --model small --epochs 250 --patience 40 --device cuda --no-amp --from-scratch --name scratch
  ```
  The scratch run needs more epochs and more patience: it learns much more
  slowly. Compare at the same model size (use `nano` for both if time is short).
  Validation loss logging is switched on automatically (the startup line says
  `validation loss logging: ON`). Re-running a command starts over; there is no
  resume option yet.

- **`sort_rfdetr_errors.py`** -- runs a trained checkpoint over a split and
  **sorts every image into folders**: `correct/` and `failure/<type>/` (wrong
  object, wrong material, wrong color, wrong material and color, same-category
  object, missed, low score, poor localization, false positive background /
  duplicate). Each image is drawn with the true boxes (green) and the model's
  (red); it also writes `image_summary.csv`, `errors_report.csv`,
  `confusions.csv`, `summary.txt` and `summary.json`.

  ```bash
  python sort_rfdetr_errors.py --dataset-dir ../../dataset --weights runs/rfdetr/pretrained
  python sort_rfdetr_errors.py --dataset-dir ../../dataset --weights runs/rfdetr/pretrained --split test --conf 0.2
  ```
  Output goes to `<run folder>/error_sort_<split>/`. `--weights` can be the run
  folder or a `.pth` file.

- **`plot_rfdetr_loss.py`** -- draws training / validation curves from the run's
  `metrics.csv`: loss per epoch (train, and validation if logged), the training
  loss components, and validation mAP50 / mAP50-95 / F1 with the best epoch
  marked. Safe to run while training is still going. Give several run folders
  to overlay them (e.g. pretrained vs scratch). Validation loss is only
  available for runs started with the current `train_rfdetr.py`; older runs
  still get the training loss and the validation mAP/F1 curves.

  ```bash
  python plot_rfdetr_loss.py runs/rfdetr/pretrained
  python plot_rfdetr_loss.py runs/rfdetr/pretrained runs/rfdetr/scratch      # saves loss_comparison.png
  ```

## Results (fill in after your runs)

Run: `______________`   Dataset split: `valid`   conf `0.3`   IoU `0.5`

| metric | value |
|---|---|
| COCO mAP50 | |
| COCO mAP50-95 | |
| precision | |
| recall | |
| F1 | |
| composite F1 (object/material/color) | |
| images fully correct | |
| objects found with exact class | |

![Loss curves](results/loss_curves.png)
*Fill in with:* `python plot_rfdetr_loss.py runs/rfdetr/pretrained --out results/loss_curves.png`
(add a second run folder to overlay pretrained vs scratch).

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

Each slot below shows 2 real examples (true boxes **green**, RF-DETR boxes **red**). They are placeholders until you fill them in:

```bash
# 1. sort the images (writes correct/ and failure/<type>/ folders)
python sort_rfdetr_errors.py --dataset-dir ../../dataset --weights runs/rfdetr/pretrained
# 2. copy 2 examples per type into this README's picture slots
python ../common/collect_examples.py --sorted-dir runs/rfdetr/pretrained/error_sort_valid --out-dir results
```

`results/examples_captions.md` is written alongside the pictures with a one-line caption for each (true class -> what the model said); paste them under the images if you want captions.

### Correct

Every object was found with the exact right class, and there are no extra boxes.

| example 1 | example 2 |
|---|---|
| ![Correct 1](results/correct/example_1.jpg) | ![Correct 2](results/correct/example_2.jpg) |

*What happened:* _fill in after the run_
Both of these predicted the object and then matched the box for each of them mostly accuratly based around where the laser was landing 

### Wrong object

A box sits on a real object but names a different object (a pen called a hammer). *Wrong class.*

| example 1 | example 2 |
|---|---|
| ![Wrong object 1](results/wrong_object/example_1.jpg) | ![Wrong object 2](results/wrong_object/example_2.jpg) |

*What happened:* _fill in after the run_
For the first wrong object it must have confused the rubber grip of the screwdriver and oranage for the rubber ducky that was blue.
Second it got the correct object but didn't match the object bounding box as much, and just generally the grey white images doesn't contain as much data
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
1. It predicted the correct result and material but it bounding box prediction for the laser was wrong because The bounding boxes weren't the best, but it could also be two overlapping predictions 
2. It predicted the wrong material aka rubber instead of the plastic that was being hit by laser
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
1. It predicted given laser and overall object that it likely guessed the object as the box, instead of the laser where it was hitting on the packing tape
2.  it saw the claw object but beause of how it was being hit guessed wrong and because of potentially some confusion with unique positioning may have confused the white plasic part and the metal from this angel
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
1. it found it but it was too small a guess
2. it gussed the class wrong and too big
### Missed: score too low

The model did find it, with the right class in the right place, but its score was under the cut-off (`--conf`, default 0.3). Lowering `--conf` would have caught it. *False negative.*

| example 1 | example 2 |
|---|---|
| ![Missed: score too low 1](results/missed_low_confidence/example_1.jpg) | ![Missed: score too low 2](results/missed_low_confidence/example_2.jpg) |

*What happened:* _fill in after the run_
1. the prediction guessed where the book hit the cover but now on the side as it went down
2. gussed both around the whole thing and indivually when it was around each two laser hits
### Poor localization

Right class, but the box overlaps the real object too little (IoU between 0.1 and 0.5). *False negative.*

| example 1 | example 2 |
|---|---|
| ![Poor localization 1](results/poor_localization/example_1.jpg) | ![Poor localization 2](results/poor_localization/example_2.jpg) |

*What happened:* _fill in after the run_
1. prediction covered some of the laser hit but truth was also overly wide
2. true labels were close but the box it put was wider
### False positive: background

A box where there is no real object. *False positive.*

| example 1 | example 2 |
|---|---|
| ![False positive: background 1](results/false_positive_background/example_1.jpg) | ![False positive: background 2](results/false_positive_background/example_2.jpg) |

*What happened:* _fill in after the run_
1. made some predictions up + shifted
2. expanded to hit a secondary laser that the labeling tru emissed
### False positive: duplicate

A second box on an object that was already found. *False positive.*

| example 1 | example 2 |
|---|---|
| ![False positive: duplicate 1](results/false_positive_duplicate/example_1.jpg) | ![False positive: duplicate 2](results/false_positive_duplicate/example_2.jpg) |

*What happened:* _fill in after the run_
1,  guessed the wrong object type
  2 guessed the wrong object type





  last run i did 
  python train_rfdetr.py --dataset-dir ..\..\dataset --model nano --epochs 100 --batch 2 --grad-accum 8 --resolution 512 --device cuda --no-amp --workers 0 --patience 20 --name nano_lowmem