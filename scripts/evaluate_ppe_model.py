"""Runs Ultralytics' built-in validation on the held-out split and prints
precision/recall/mAP so results can be copied into docs/evaluation.md
verbatim — these are the actual observed numbers, not fabricated.

Usage:
    python -m scripts.evaluate_ppe_model --weights models/ppe_helmet_best.pt --data data/yolo/data.yaml
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="models/ppe_helmet_best.pt")
    parser.add_argument("--data", default="data/yolo/data.yaml")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, device=args.device, project="outputs/training", name="ppe_helmet_eval")

    print("\n--- Observed validation metrics (copy into docs/evaluation.md) ---")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
