# Model Setup

Two YOLOv8n checkpoints are used by this project. Neither is committed to
Git (see root `.gitignore`) — both are either auto-downloaded or produced
locally by a script.

## 1. Person detector (`yolov8n.pt`)

- **Source:** Ultralytics, pretrained on COCO
- **License:** AGPL-3.0 (Ultralytics) — see https://github.com/ultralytics/ultralytics
- **Provenance:** we did **not** train this. It is downloaded automatically
  by the `ultralytics` package the first time `src.detection.detector.PersonDetector`
  is instantiated, and cached by `ultralytics` outside the repo.
- Used only for its existing `person` class.

## 2. PPE / helmet detector (`models/ppe_helmet_best.pt`)

- **Source:** YOLOv8n, **fine-tuned by us** via transfer learning on the
  Kaggle "Safety Helmet Detection" dataset (see `data/README.md`).
- Produced by running:
  ```
  python -m scripts.prepare_dataset
  python -m scripts.train_ppe_model --data data/yolo/data.yaml --epochs 8 --imgsz 416 --batch 8 --device cpu
  ```
  which writes the best checkpoint to `models/ppe_helmet_best.pt`. The
  actual training run on this machine completed 3 of the requested 8 epochs
  before being stopped by an OS/harness memory guard (this laptop has
  limited RAM headroom for CPU-only training) — see `docs/evaluation.md`
  for the exact reason and the resulting metrics.
- Actual observed validation metrics from our training run are recorded in
  `docs/evaluation.md` (not fabricated, not the published YOLOv8 COCO numbers).
