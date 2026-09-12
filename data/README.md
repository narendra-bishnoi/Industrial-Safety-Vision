# Dataset Setup

This project does not ship any dataset in the repository (see the root
`.gitignore`) — the license permits redistribution (CC0), but committing a
5000-image dataset to Git is bad practice and unnecessary.

## Source

- **Name:** Safety Helmet Detection
- **Author/host:** andrewmvd, Kaggle
- **URL:** https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
- **License:** CC0 1.0 (public domain)
- **Size:** 5000 images, 25,502 annotated objects, Pascal VOC XML format
- **Classes used by this project:** `helmet`, `head`, `person`

## Steps

1. Download the zip from the Kaggle URL above (requires a free Kaggle account).
2. Extract it so you have:
   ```
   data/raw/images/*.png
   data/raw/annotations/*.xml
   ```
3. Convert to YOLO format and create the train/val split:
   ```
   python -m scripts.prepare_dataset --source data/raw --dest data/yolo --val-split 0.15
   ```
4. Fine-tune YOLOv8n on the prepared data (see root README for full command):
   ```
   python -m scripts.train_ppe_model --data data/yolo/data.yaml --epochs 8 --imgsz 416 --batch 8 --device cpu
   ```
   Note: on the machine used for this submission, training was interrupted
   by an OS/harness memory guard after 3 of the requested 8 epochs
   completed — see `docs/evaluation.md` for the actual results and cause.

`data/raw/` and `data/yolo/` are gitignored — they are regenerated from the
steps above, not committed.
