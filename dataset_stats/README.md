# dataset_stats

[← back to the main README](../README.md)

Audits a YOLO-format dataset (**reads only; changes nothing**): class balance, box sizes, label problems, and whether validation images are near-copies of training images (leakage).

## Run it

From this folder (`dataset_stats/`; the dataset is one level up):

```bash
# Windows: double-click or run
run_stats.bat                          # uses ..\dataset
run_stats.bat C:\path\to\dataset

# any OS
python dataset_stats.py --dataset-dir ../dataset --out dataset_stats_out
```

Needs `pip install pillow numpy pyyaml matplotlib` (matplotlib only for the charts). Class names come from `data.corrected.yaml` if present, else `data.yaml`.

## Output (`dataset_stats_out/`)

- `dataset_stats_report.txt` - the readable report
- `per_class.csv`, `per_object.csv`, `per_material.csv`, `per_color.csv`
- `label_issues.csv` - every broken/odd label with file and line
- `duplicates_exact.csv`, `train_valid_near_duplicates.csv` - leakage check
- PNG charts: images per class, instances per object/material/color, box sizes

A copy of the latest run is committed in [`../stats_after_split/`](../stats_after_split/).

## Options

| flag | meaning |
|---|---|
| `--min-train-images 30`, `--min-valid-images 5` | Flag classes with fewer images than this |
| `--near-dup-distance 6` | Max image-fingerprint distance (0-64) counted as a near-duplicate |
| `--no-hash` | Skip the duplicate check (much faster) |
| `--no-pixel-stats`, `--pixel-sample 600` | Skip / limit brightness and sharpness sampling |
| `--no-plots` | Skip the charts |
| `--data FILE` | Use a specific class-name YAML |

## Latest findings (`stats_after_split/`)

- **47 classes**, 1,365 train / 340 valid images (about 20% valid), ~1 labelled object per image; 1,720 boxes in total; no label problems found.
- **Class imbalance:** between 2 and 63 train instances per class (median 26, 31.5x spread); 24 of 47 classes get a "needs attention" warning. Fewest train images: `insulated water bottle` (2).
- **Boxes are small:** the median box is about 10% of the image side, so keep the training resolution high enough (`--imgsz 640` for YOLO; RF-DETR `--resolution`).
- **Train/valid leakage:** 188 of 340 validation images (55%) have a near-duplicate in train (56 look identical); no exact duplicates. Validation scores are probably optimistic. To test on unseen objects/sessions use `training_scripts/common/resplit_dataset.py --mode object-disjoint`.
