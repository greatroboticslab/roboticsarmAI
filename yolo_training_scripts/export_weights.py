#!/usr/bin/env python3
"""
export_weights.py

Exports a trained YOLO checkpoint (best.pt) to a deployment format --
ONNX by default, since that's the most portable choice for running
inference outside of Python/ultralytics (e.g. from the robotic arm's own
inference stack), but any format ultralytics supports can be requested.

This is a separate, later step from training/evaluation: train_yolo.py
produces best.pt, evaluate.py tells you how good it is, and once you're
happy with it, this script turns it into something you can actually deploy.

Common formats:
    onnx          portable, works with ONNX Runtime, OpenCV DNN, etc. (default)
    torchscript   for a pure-PyTorch deployment environment
    engine        NVIDIA TensorRT (GPU-only, must be exported on the target GPU type)
    openvino      Intel CPUs/VPUs
    coreml        Apple devices
    tflite        mobile/edge (Android, microcontrollers via LiteRT)
    saved_model   TensorFlow SavedModel

Usage:
    # plain ONNX export (most common choice for deployment)
    python export_weights.py --weights runs/detect/runs/train/exp/weights/best.pt

    # simplified, dynamic-batch ONNX, matching the imgsz you trained at
    python export_weights.py --weights .../best.pt --imgsz 640 --dynamic --simplify

    # FP16 (half precision) TensorRT engine for an NVIDIA GPU target
    python export_weights.py --weights .../best.pt --format engine --quantize 16 --device 0

    # INT8 quantization needs representative calibration images -- point --data at your dataset
    python export_weights.py --weights .../best.pt --format onnx --quantize 8 --dataset-dir ../dataset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from yolo_common import resolve_data_yaml


def export_weights(
    weights: Path,
    format: str = "onnx",
    imgsz: int = 640,
    batch: int = 1,
    quantize: int | None = None,
    dynamic: bool = False,
    simplify: bool = False,
    opset: int | None = None,
    nms: bool | None = None,
    device: str | None = None,
    data_yaml: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install it with:\n    pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(weights))
    export_kwargs = dict(format=format, imgsz=imgsz, batch=batch, dynamic=dynamic, simplify=simplify)
    if quantize is not None:
        export_kwargs["quantize"] = quantize
    if opset is not None:
        export_kwargs["opset"] = opset
    if nms is not None:
        export_kwargs["nms"] = nms
    if device is not None:
        export_kwargs["device"] = device
    if data_yaml is not None:
        export_kwargs["data"] = str(data_yaml)

    print(f"Exporting {weights} -> format={format} imgsz={imgsz} batch={batch}"
          f"{f' quantize={quantize}' if quantize is not None else ''}"
          f"{' dynamic' if dynamic else ''}{' simplify' if simplify else ''} ...")

    result_path = Path(model.export(**export_kwargs))
    print(f"Export produced: {result_path}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if result_path.is_dir():
            dest = output_dir / result_path.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(result_path, dest)
        else:
            dest = output_dir / result_path.name
            shutil.copy2(result_path, dest)
        print(f"Copied to: {dest}")
        result_path = dest

    size_note = ""
    if result_path.is_file():
        size_mb = result_path.stat().st_size / (1024 * 1024)
        size_note = f" ({size_mb:.1f} MB)"
    print(f"\nFinal exported model path: {result_path}{size_note}")
    return result_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True, help="Path to a trained .pt checkpoint (e.g. best.pt)")
    ap.add_argument("--format", default="onnx", help="Export format: onnx (default), torchscript, engine, openvino, coreml, saved_model, tflite, etc.")
    ap.add_argument("--imgsz", type=int, default=640, help="Input size to bake into the export -- match what you trained/evaluated at")
    ap.add_argument("--batch", type=int, default=1, help="Fixed batch size for the export (default: 1, typical for single-image deployment inference)")
    ap.add_argument("--quantize", type=int, default=None, choices=[8, 16], help="16 for FP16 (half precision), 8 for INT8 (needs --dataset-dir for calibration images). Omit for full FP32.")
    ap.add_argument("--dynamic", action="store_true", help="Allow dynamic input shapes/batch size (not supported by every format)")
    ap.add_argument("--simplify", action="store_true", help="Simplify the exported graph (ONNX and a few others)")
    ap.add_argument("--opset", type=int, default=None, help="ONNX opset version (default: let ultralytics choose)")
    ap.add_argument("--nms", action="store_true", help="Embed NMS in the exported model where supported, instead of leaving raw output")
    ap.add_argument("--device", default=None, help="e.g. 0, cpu. Needed for some formats (e.g. TensorRT 'engine' must be exported on the target GPU type)")
    ap.add_argument("--dataset-dir", type=Path, default=None, help="Path to dataset/ -- only needed for INT8 quantization calibration images")
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml for calibration, overrides --dataset-dir auto-detection")
    ap.add_argument("--output-dir", type=Path, default=None, help="Copy the exported file/folder here afterward (default: leave it where ultralytics wrote it, next to the weights)")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights file not found: {args.weights}")

    data_yaml = None
    if args.quantize == 8 or args.data is not None:
        if args.data is not None or args.dataset_dir is not None:
            data_yaml = resolve_data_yaml(args.dataset_dir, args.data)
        elif args.quantize == 8:
            print("[warn] --quantize 8 (INT8) works best with calibration images -- pass --dataset-dir or --data "
                  "so ultralytics can use your real images to calibrate, otherwise it may fall back to synthetic data.")

    export_weights(
        weights=args.weights,
        format=args.format,
        imgsz=args.imgsz,
        batch=args.batch,
        quantize=args.quantize,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        nms=args.nms if args.nms else None,
        device=args.device,
        data_yaml=data_yaml,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
