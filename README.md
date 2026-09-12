# Industrial Safety Vision System

A command-line Computer Vision pipeline that analyzes images or video from
an industrial/construction environment, detects people, evaluates helmet
PPE compliance through spatial reasoning (not a blind classifier flag),
tracks people across video frames, and produces evidence-backed
machine-readable and human-readable safety reports.

Academic submission (VITyarthi). See `statement.md` for the concise problem
statement/scope, and `docs/` for architecture, workflow, and evaluation
detail.

## Overview

Given an image or video, the system:

1. Validates and preprocesses the input.
2. Detects people (general-purpose COCO-pretrained YOLOv8n).
3. Detects head-PPE state — `helmet` or bare `head` (YOLOv8n fine-tuned by
   us on a public dataset).
4. **Associates** each detected person with nearby PPE detections in their
   head region (spatial reasoning, not a global count).
5. Applies a rule engine to classify compliant vs. violation, distinguishing
   a high-confidence violation (bare head positively detected) from a
   weaker, unverified one (nothing detected in the head region at all).
6. Tracks people across video frames with a lightweight IoU tracker so the
   same person isn't reported as a new violation every frame.
7. Optionally checks whether a person has entered a restricted zone
   (config-defined polygon).
8. Captures an annotated evidence image for every violation.
9. Writes a JSON report, an optional CSV, and prints a terminal summary.

## Problem Statement

See `statement.md`.

## Objectives

- Demonstrate a defensible, explainable CV pipeline (not "if class not
  found: violation").
- Use an appropriately small, reproducible model given laptop-class
  hardware (RTX 2050, 4GB VRAM, or CPU-only).
- Keep detection, inference/association, and rule-based decision explicitly
  separate, testable, and documented.
- Report only measured results — no fabricated accuracy numbers.

## Functional Requirements

1. **Input & Preprocessing** (`src/preprocessing/`) — accept an image or
   video file, validate it exists and has a supported extension, decode and
   resize frames.
2. **Detection** (`src/detection/`) — detect people (COCO-pretrained
   YOLOv8n) and head-PPE state (`helmet`/`head`, fine-tuned YOLOv8n).
3. **Safety Rule Engine** (`src/safety/`) — associate each person with
   nearby PPE detections and classify compliant vs. violation; separately
   check restricted-zone entry.
4. **Tracking** (`src/tracking/`) — assign stable IDs to people across
   video frames so one person isn't reported as repeat new violations.
5. **Evidence Extraction** (`src/evidence/`) — capture an annotated image +
   metadata for every violation event.
6. **Reporting** (`src/reporting/`) — produce a JSON report, optional CSV,
   and a human-readable terminal summary.

(6 modules — exceeds the assignment's minimum of 3.)

## Non-Functional Requirements

- **Performance** — inference time and FPS are measured and reported on
  every run (see `docs/evaluation.md` for real, measured numbers).
- **Reliability** — invalid/missing/corrupt inputs, bad config values, and
  missing model weights all raise clear, typed exceptions instead of
  crashing or silently producing wrong output (`src/config.py`,
  `src/preprocessing/processor.py`, `src/detection/detector.py`).
- **Usability** — a documented CLI with `--help`, sensible defaults, and a
  readable terminal summary; no GUI required for the core workflow.
- **Maintainability** — modular package structure, config separated from
  logic, type hints, no module depends on `ultralytics` except
  `src/detection/detector.py` (see `docs/architecture.md`).
- **Error Handling & Logging** — every module raises typed exceptions at
  its boundary; `src/utils/logger.py` provides structured console + rotating
  file logging used throughout `src/main.py`.

(5 non-functional requirements — exceeds the assignment's minimum of 4.)

## Features

- CLI for image and video input with input validation and clear errors.
- Two-model detection pipeline (general person detector + specialist PPE detector).
- Explainable person–PPE spatial association.
- IoU-based multi-object tracking.
- Restricted-zone polygon check.
- Evidence capture per violation (annotated frame + metadata).
- JSON, CSV, and terminal reporting.
- Structured logging to console and a rotating log file.
- Unit-tested rule engine, tracker, config validation, and reporting —
  independent of live model inference.

## Computer Vision Concepts

- Image/video I/O and preprocessing (decode, resize, color handling) — OpenCV
- Object detection: single-stage anchor-free detector (YOLOv8), confidence
  thresholding, non-max suppression (handled internally by Ultralytics,
  configured explicitly via our confidence parameter)
- Transfer learning / fine-tuning a pretrained detector on a small custom dataset
- Spatial reasoning: bounding-box containment for person–PPE association
- Multi-object tracking: IoU-based frame-to-frame data association
- Detection evaluation: precision, recall, mAP, F1, inference latency/FPS
- Computational geometry: point-in-polygon (ray casting) for zone checks

## System Architecture

See `docs/architecture.md` for the full pipeline diagram and module
responsibility table.

```
Input → Preprocessing → [Person Detector, PPE Detector] → Tracker
      → Rule Engine (association + decision) → Zone Check
      → Evidence Extractor → Reporter (JSON/CSV/text) + Annotated output
```

## Project Structure

```
project-root/
├── README.md
├── statement.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── main.py                CLI entry point / orchestration
│   ├── config.py               AppConfig, ZoneConfig, validation
│   ├── preprocessing/processor.py   input loading, validation, frame extraction
│   ├── detection/detector.py        PersonDetector, PPEDetector (YOLOv8n)
│   ├── tracking/tracker.py          IoU-based multi-object tracker
│   ├── safety/rule_engine.py        person-PPE association + violation rules
│   ├── safety/zones.py              restricted-zone polygon check
│   ├── evidence/extractor.py        evidence image + metadata capture
│   ├── reporting/reporter.py        JSON/CSV/text report generation
│   └── utils/logger.py              logging setup
├── scripts/
│   ├── prepare_dataset.py      Kaggle VOC XML -> YOLO format converter
│   ├── train_ppe_model.py      fine-tunes YOLOv8n on the prepared dataset
│   └── evaluate_ppe_model.py   runs validation, prints real metrics
├── tests/                      pytest unit tests (no live model needed)
├── data/README.md              dataset source, license, setup steps
├── models/README.md            model provenance, setup steps
├── outputs/                    annotated media, evidence, reports, logs (gitignored)
└── docs/
    ├── architecture.md         pipeline + module diagrams
    ├── workflow.md              sequence, use-case, class diagrams
    └── evaluation.md            evaluation methodology + results
```

## Requirements

- Python 3.10+ (developed and tested on 3.14)
- pip
- ~2GB free disk for dependencies (PyTorch is the largest)
- Optional: NVIDIA GPU + CUDA for faster inference (`--device cuda`); CPU
  works for images and short video clips

## Environment Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

## Dependency Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` is intentionally short — `ultralytics` pulls in `torch`/
`torchvision` as transitive dependencies, and no library is listed that
isn't actually used in the pipeline (no `pandas`/`matplotlib`, since the
project doesn't need a dataframe or a plot to function).

## Dataset Setup

See `data/README.md`. Short version: download the Kaggle "Safety Helmet
Detection" dataset (CC0 license), extract into `data/raw/`, then:

```bash
python -m scripts.prepare_dataset --source data/raw --dest data/yolo --val-split 0.15
```

## Model Setup

See `models/README.md`. The person detector auto-downloads on first run.
The PPE detector must be fine-tuned once:

```bash
python -m scripts.train_ppe_model --data data/yolo/data.yaml --epochs 8 --imgsz 416 --batch 8 --device cpu
python -m scripts.evaluate_ppe_model --weights models/ppe_helmet_best.pt --data data/yolo/data.yaml
```

This writes `models/ppe_helmet_best.pt`, which `src/main.py` loads by
default.

## Configuration

Restricted zones are defined in a JSON file, e.g. `zones.json`:

```json
{
  "zones": [
    { "name": "crane_swing_area", "points": [[100, 100], [500, 100], [500, 400], [100, 400]] }
  ]
}
```

Pass it with `--zones zones.json`. Other run parameters (confidence
threshold, model paths, device) are CLI flags — see below.

## Command-Line Execution

```bash
python -m src.main --input <path> --mode {image,video} --output outputs/ [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | Path to image or video file |
| `--output` | `outputs` | Output directory |
| `--mode` | auto-detected from extension | `image` or `video` |
| `--confidence` | `0.4` | Detection confidence threshold |
| `--person-model` | `yolov8n.pt` | Person detector weights |
| `--ppe-model` | `models/ppe_helmet_best.pt` | PPE detector weights |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--save-evidence` / `--no-save-evidence` | on | Save evidence images |
| `--zones` | none | Path to restricted-zone JSON |
| `--report` | `json,text` | Comma-separated: `json`, `csv`, `text` |
| `--frame-stride` | `1` | Process every Nth video frame |

## Example Commands

```bash
python -m src.main --input data/sample.jpg --mode image --output outputs/
python -m src.main --input data/sample.mp4 --mode video --output outputs/ --confidence 0.5 --report json,csv,text
python -m src.main --input data/sample.mp4 --mode video --zones zones.json --device cuda
```

## Expected Output

- `outputs/annotated_<name>.jpg` or `.mp4` — input with person boxes/IDs drawn.
- `outputs/evidence/frame######_track#_<TYPE>.jpg` — one image per violation event.
- `outputs/report.json` — full structured run report.
- `outputs/violations.csv` — one row per violation (if `csv` requested).
- `outputs/logs/run.log` — rotating log file.
- A terminal summary (frame count, persons tracked, violations by type, FPS).

## Testing

```bash
pytest -v
```

Tests cover: invalid input handling (`test_preprocessing.py`,
`test_validation.py`), safety rule/zone logic (`test_rules.py`), tracker ID
persistence (`test_tracker.py`), and report generation
(`test_reporting.py`). None of these require the trained PPE weights or a
GPU — model inference is isolated behind `PersonDetector`/`PPEDetector` and
tested indirectly via synthetic detections. All 29 tests pass.

**End-to-end verification performed** (real CLI runs, not just unit tests):
image-mode run on a real sample image producing correct person detection,
`NO_HELMET` violations, evidence images, and `report.json`; a second run
with `--zones` confirming `RESTRICTED_ZONE_ENTRY` fires correctly alongside
PPE violations. **Video mode was not end-to-end tested** — no sample video
file was available for this submission (video *support* exists in the code
and shares the same tested rule/tracker/reporting logic, but hasn't itself
been run). The `NO_HELMET_UNVERIFIED` branch was not exercised in a live
run either — it's covered by a deterministic unit test instead (see
`docs/evaluation.md` for details on both).

## Evaluation Methodology

See `docs/evaluation.md`. Summary: the person detector's accuracy is cited
from Ultralytics' published COCO benchmark (not measured by us); the PPE
detector's accuracy is measured directly from our own fine-tuning run via
`scripts/evaluate_ppe_model.py`; the rule engine is verified with
deterministic unit tests, not accuracy metrics.

## Limitations

- **PPE model trained for 3 epochs, not the configured 8** — the training
  process on this machine was terminated by an OS/harness memory guard
  after 3 epochs completed (limited free RAM on this laptop for CPU-only
  training, not a code defect). Metrics were still trending upward at
  interruption (mAP50: 0.561 → 0.583 → 0.596 across the 3 completed
  epochs), so more training would likely help further — see
  `docs/evaluation.md` for the full, real numbers and per-class breakdown.
- Helmet compliance only (no vest/other PPE in this submission).
- PPE detector accuracy is bounded by a 5000-image dataset — expect misses
  on unusual angles, lighting, or helmet colors not well represented in it.
  The dataset's own `person` class is too sparse to be usable (mAP50 0.022
  in our evaluation) — this project deliberately does not rely on it.
- Greedy IoU tracker can lose/swap IDs under heavy occlusion or fast motion.
- Restricted zones are 2D image-space polygons, not real-world calibrated coordinates.
- `NO_HELMET_UNVERIFIED` events reflect *absence of detection*, not a
  confirmed violation — always human-reviewable via the evidence image.
  This branch was verified with a deterministic unit test, not a live run.
- Video mode was not end-to-end tested for this submission (no sample video
  file was available) — see Testing above.

## Ethical / Privacy Considerations

This system processes footage of real or simulated people. It is designed
as a **review aid for human safety personnel**, not an autonomous
enforcement or disciplinary tool — every flagged event includes an evidence
image specifically so it can be human-verified before any action is taken.
It performs no facial recognition, identity inference, or biometric
matching of any kind — only anonymous bounding boxes and a per-run track
ID that resets every run. Footage of real individuals should only be used
with appropriate consent/authorization from the site and recorded persons;
this repository ships no personal data. False positives and false
negatives are expected (see Limitations) and are a core reason this stays
assistive rather than automated.

## References

- Ultralytics YOLOv8 — https://docs.ultralytics.com/models/yolov8/ (detection framework, pretrained COCO weights, training/validation API), AGPL-3.0 license
- Kaggle "Safety Helmet Detection" dataset, andrewmvd — https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection, CC0 1.0 license
- OpenCV — https://opencv.org/ (image/video I/O, drawing)
- Bochinski, E., Eiselein, V., Sikora, T. (2017). "High-Speed Tracking-by-Detection Without Using Image Information." *AVSS 2017* — conceptual basis for the IoU-tracking approach implemented independently in `src/tracking/tracker.py`

## License

This project's own code is MIT-licensed (see `LICENSE`). It depends on
Ultralytics (AGPL-3.0) and a CC0-licensed dataset — see above.
