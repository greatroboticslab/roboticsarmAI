# dataset_stats

Audits a YOLO-format dataset (reads only; changes nothing).

    run_stats.bat                      (uses ..\..\dataset)
    run_stats.bat C:\path\to\dataset
    python dataset_stats.py --dataset-dir ..\..\dataset --out dataset_stats_out

Needs `pip install pillow numpy pyyaml matplotlib` (matplotlib only for the charts).

Output folder `dataset_stats_out`:
- `dataset_stats_report.txt` – the readable report
- `per_class.csv`, `per_object.csv`, `per_material.csv`, `per_color.csv`
- `label_issues.csv` – every broken/odd label with file and line
- `duplicates_exact.csv`, `train_valid_near_duplicates.csv` – leakage check
- PNG charts: images per class, instances per object/material/color, box sizes

Useful options: `--min-train-images 30`, `--min-valid-images 5`, `--near-dup-distance 6`,
`--no-hash` (skip duplicate check, much faster), `--no-pixel-stats`, `--no-plots`.
