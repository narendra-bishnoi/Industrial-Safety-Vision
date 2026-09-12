"""Fine-tunes YOLOv8n on the prepared PPE dataset (see prepare_dataset.py).

This is the training run we actually execute and report real metrics from —
not a published benchmark presented as our own result.

Usage:
    python -m scripts.train_ppe_model --data data/yolo/data.yaml --epochs 50 --device cpu
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/yolo/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", default="models/ppe_helmet_best.pt")
    args = parser.parse_args()

    if not Path(args.data).is_file():
        raise SystemExit(f"Dataset config not found: {args.data}. Run scripts/prepare_dataset.py first.")

    model = YOLO("yolov8n.pt")  # start from COCO-pretrained weights (transfer learning)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="outputs/training",
        name="ppe_helmet",
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, args.out)
    print(f"Best weights copied to {args.out}")
    print("Run scripts/evaluate_ppe_model.py to record precision/recall/F1/mAP for docs/evaluation.md")


if __name__ == "__main__":
    main()
