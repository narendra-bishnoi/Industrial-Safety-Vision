# Evaluation Methodology

## Person detector (YOLOv8n, COCO-pretrained)

We did not train this model. Its accuracy is reported by Ultralytics on the
COCO val2017 benchmark in their official documentation/model zoo
(https://docs.ultralytics.com/models/yolov8/) — cited as **their** published
number, not measured by us. Our own evaluation of this component is
qualitative: does it reliably localize people in our sample images/video.

## PPE detector (YOLOv8n, fine-tuned by us)

Measured with `scripts/evaluate_ppe_model.py`, which runs Ultralytics'
standard validation loop against the held-out 15% split created by
`scripts/prepare_dataset.py`. Metrics reported: Precision, Recall, mAP@50,
mAP@50-95.

**Status: run complete.** Numbers below are the actual printed output of
`python -m scripts.evaluate_ppe_model --weights models/ppe_helmet_best.pt
--data data/yolo/data.yaml --device cpu`, run on this machine — not
estimated, not the published YOLOv8/COCO numbers.

| Metric | Value (all classes) |
|---|---|
| Precision | 0.9200 |
| Recall | 0.5382 |
| mAP@50 | 0.5965 |
| mAP@50-95 | 0.3401 |
| Training images / val images | 4250 / 750 |
| Epochs actually completed | 3 (training config requested 8; see note below) |
| CPU inference speed | 64.4 ms/image (≈15.5 FPS), measured during this evaluation run |

**Per-class breakdown** (this is the more informative table — the combined
row above is pulled down heavily by one class we don't actually rely on):

| Class | Images (val) | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| helmet | 688 | 2945 | 0.905 | 0.813 | 0.897 | 0.507 |
| head | 138 | 895 | 0.855 | 0.802 | 0.870 | 0.503 |
| person | 24 | 93 | 1.000 | 0.000 | 0.022 | 0.010 |

The `person` class essentially failed to learn (only 93 labeled instances
across 24 validation images) — this is expected, not a bug, and it
retroactively confirms the design decision documented in
`docs/architecture.md`: this project does not use this dataset's `person`
class at inference time at all, precisely because it is too sparse to train
reliably. The classes the pipeline actually depends on — `helmet` and
`head` — both reach mAP50 ≈ 0.87-0.90 after only 3 epochs of fine-tuning,
which is a reasonable, defensible result for an academic submission.

**Note on epoch count:** training was configured for 8 epochs
(`--epochs 8 --imgsz 416 --batch 8 --device cpu`) but the process was
terminated by the OS/harness after 3 completed epochs due to system memory
pressure on this machine (a repeatable issue on this particular laptop, not
a bug in the training code — see Limitations). Per-epoch validation showed
a consistent improving trend through all 3 completed epochs (mAP50: 0.561 →
0.583 → 0.596), so more epochs would likely improve results further; this
is noted as future work rather than claimed as already done.

## Rule engine

Deterministic logic, evaluated with unit tests (`tests/test_rules.py`)
against synthetic bounding boxes rather than model output — this isolates
"is the association/decision logic correct" from "is the detector
accurate," which the assignment explicitly asks us to keep separate.

## Runtime performance (FPS)

`RunStats.inference_seconds` / `fps` is computed and printed on every real
run (see `src/reporting/reporter.py`). Two real, measured data points exist:

- End-to-end CLI run on one real image (`data/raw/images/hard_hat_workers0.png`,
  both detectors + tracker + rule engine + zone check + evidence capture,
  CPU): **5.47s for 1 frame (0.18 FPS)**.
- PPE detector alone, measured by `scripts/evaluate_ppe_model.py` over the
  750-image val set (CPU): **64.4 ms/image (≈15.5 FPS)**.

No video has been run through the pipeline (see below) — video-mode FPS is
therefore not reported, since a single-image number would not represent
sustained per-frame video throughput.

## What we are not claiming

- No claim of state-of-the-art accuracy.
- No claim that `NO_HELMET_UNVERIFIED` events are confirmed violations —
  they are flagged precisely because the evidence is weaker (no positive
  detection either way in the head region).
- **No video was run through the pipeline for this submission.** Only
  image-mode end-to-end runs were performed (see above and
  `README.md#testing`). Video *support* exists in the code
  (`VideoFrameReader`, `--mode video`) and is covered by the same rule
  engine/tracker/reporting logic already tested, but it has not itself
  been exercised end-to-end with a real video file.
- **The `NO_HELMET_UNVERIFIED` branch was not exercised in a live CLI run**
  (that would require an image where the model misses both `helmet` and
  `head` in the head region, which was not deliberately constructed). It
  is covered by a deterministic unit test
  (`tests/test_rules.py::test_no_detection_in_head_region_is_unverified_violation`),
  which is sufficient because that branch is pure rule logic with no model
  dependency.
- No mAP figure for the *combined* end-to-end pipeline (person+PPE+rule),
  since there is no ground-truth "violation" labelled video available to
  us — this would require hand-labeling test footage, which is out of
  scope for this submission and is stated here rather than glossed over.
