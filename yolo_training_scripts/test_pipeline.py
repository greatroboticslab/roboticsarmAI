#!/usr/bin/env python3
"""
test_pipeline.py

A smoke test for the whole train -> evaluate pipeline. Runs a very short
training job (default: 1 epoch, small image size/batch) and then runs
evaluate.py's validation pass against it, so you can confirm:

  - dataset/data.corrected.yaml (or data.yaml) is well-formed and loads
  - ultralytics can actually train and produce a checkpoint on this dataset
  - evaluate.py runs end-to-end and produces a report with real numbers

This is NOT a measure of model quality -- 1-2 epochs will produce poor
metrics no matter what. It only tells you the pipeline runs cleanly, so a
real, long training run doesn't fail after hours on something a 2-minute
test would have caught (a malformed data.yaml, a bad image, a broken class
id, evaluate.py crashing on this ultralytics version, etc).

By default everything is written under a throwaway directory
(runs/_pipeline_test/<timestamp>) and left there for you to inspect;
pass --cleanup to delete it after a successful run.

Usage:
    python test_pipeline.py --dataset-dir ../../dataset
    python test_pipeline.py --dataset-dir ../../dataset --epochs 3 --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from yolo_common import resolve_data_yaml


def check(condition: bool, message: str, failures: list[str]):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        failures.append(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml, overrides --dataset-dir auto-detection")
    ap.add_argument("--model", default="yolov8n.pt", help="Base checkpoint to start from (default: yolov8n.pt, the smallest/fastest)")
    ap.add_argument("--epochs", type=int, default=1, help="Kept tiny on purpose -- this is a smoke test, not a real training run")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu, mps. Default: let ultralytics auto-select")
    ap.add_argument("--project", default="runs/_pipeline_test", help="Throwaway directory for this test run")
    ap.add_argument("--cleanup", action="store_true", help="Delete the test run's output directory afterward")
    args = ap.parse_args()

    failures: list[str] = []

    print("=" * 70)
    print("PIPELINE SMOKE TEST")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install it with:\n    pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    # 1. data.yaml resolves and loads
    print("\n[1/4] Resolving dataset config...")
    try:
        data_yaml = resolve_data_yaml(args.dataset_dir, args.data)
        from yolo_common import load_data_yaml
        data = load_data_yaml(data_yaml)
        nc = data.get("nc", len(data.get("names", [])))
        check(True, f"loaded {data_yaml} (nc={nc})", failures)
    except Exception as e:
        check(False, f"could not resolve/load a data.yaml: {e}", failures)
        _finish(failures, args, run_dir=None)
        return

    # 2. a tiny training run completes and produces a checkpoint
    run_label = f"test_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"\n[2/4] Running a {args.epochs}-epoch smoke-test training job (imgsz={args.imgsz}, batch={args.batch})...")
    print("       (this is only to prove the pipeline works -- ignore the metrics)")
    try:
        model = YOLO(args.model)
        train_kwargs = dict(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=run_label,
            verbose=False,
        )
        if args.device is not None:
            train_kwargs["device"] = args.device
        train_results = model.train(**train_kwargs)
        # Don't assume where ultralytics saved things -- ask it. Depending on
        # version/global settings, the real save_dir can differ from a naive
        # Path(project)/name guess (e.g. some versions nest an extra runs/<task>/
        # prefix in front of a relative --project).
        run_dir = Path(getattr(train_results, "save_dir", Path(args.project) / run_label))
        best_weights = run_dir / "weights" / "best.pt"
        check(best_weights.exists(), f"training produced a checkpoint at {best_weights}", failures)
    except Exception as e:
        check(False, f"training raised an exception: {e}", failures)
        _finish(failures, args, run_dir=None)
        return

    # 3. evaluate.py runs against the val split and returns real numbers
    print("\n[3/4] Running evaluate.py against the validation split...")
    try:
        from evaluate import run_evaluation
        result = run_evaluation(
            weights=best_weights,
            data_yaml=data_yaml,
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(run_dir),
            name="eval_val",
        )
        overall = result["overall"]
        has_numbers = all(k in overall for k in ("precision", "recall", "mAP50", "mAP50-95"))
        check(has_numbers, "evaluation produced precision/recall/mAP50/mAP50-95", failures)
        check(len(result["per_class"]) > 0, f"per-class report has {len(result['per_class'])} row(s)", failures)
        print(f"       (for reference only, not a quality signal at {args.epochs} epoch(s)): "
              f"mAP50={overall['mAP50']:.4f}  mAP50-95={overall['mAP50-95']:.4f}")
    except Exception as e:
        check(False, f"evaluate.py raised an exception: {e}", failures)

    # 4. if a test split exists, make sure that path works too
    if (args.dataset_dir / "test" / "images").exists():
        print("\n[4/4] Test split detected -- running evaluate.py against it too...")
        try:
            from evaluate import run_evaluation
            result = run_evaluation(
                weights=best_weights,
                data_yaml=data_yaml,
                split="test",
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=str(run_dir),
                name="eval_test",
            )
            check("overall" in result, "evaluation against the test split completed", failures)
        except Exception as e:
            check(False, f"evaluate.py raised an exception on the test split: {e}", failures)
    else:
        print("\n[4/4] No test split found under dataset/test/ -- skipped (this is fine, it's optional).")

    _finish(failures, args, run_dir)


def _finish(failures: list[str], args, run_dir: Path | None):
    print("\n" + "=" * 70)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED")
        for f in failures:
            print(f"  - {f}")
        print("=" * 70)
        if run_dir is not None:
            print(f"Test run artifacts kept for debugging at: {run_dir}")
        sys.exit(1)

    print("RESULT: all checks PASSED -- the train -> evaluate pipeline runs cleanly.")
    print("=" * 70)
    if run_dir is not None and args.cleanup:
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"Cleaned up test run directory: {run_dir}")
    elif run_dir is not None:
        print(f"Test run artifacts (including the eval report) left at: {run_dir}")
        print("Pass --cleanup next time to delete them automatically after a passing run.")


if __name__ == "__main__":
    main()
