# Problem Statement

Manual visual monitoring of Personal Protective Equipment (PPE) compliance
on industrial and construction sites is labor-intensive, inconsistent
across observers, and cannot scale to continuous coverage of every camera
feed. This project builds a Computer Vision pipeline that processes
recorded images or video from such environments, detects people, evaluates
whether each detected person's head region shows helmet-compliant PPE
using spatial reasoning (not blind classification), tracks people across
video frames, and produces evidence-backed violation reports for human
safety personnel to review.

# Project Scope

- Input: a single image or a video file (webcam optional, not required for
  core submission).
- Detection: person localization (COCO-pretrained YOLOv8n) + helmet/bare-head
  classification (YOLOv8n fine-tuned by us on a public dataset).
- Decision: an explicit, testable rule engine that associates each person
  with nearby PPE detections before deciding compliant vs. violation.
- Tracking: a lightweight IoU-based tracker so one person isn't counted as
  multiple violations across consecutive frames of video.
- Output: annotated image/video, evidence images per violation, and a
  JSON + text (+ optional CSV) report.
- Optional restricted-zone check via a config-defined polygon.

# Target Users

Safety officers and site supervisors reviewing recorded footage, and — in
an academic context — an evaluator assessing the CV pipeline design,
implementation, and reasoning. The system is a **review aid**, not an
autonomous enforcement tool.

# High-Level Features

1. CLI-driven image/video processing with input validation.
2. Person + PPE object detection using an appropriate pretrained/fine-tuned model.
3. Person–PPE spatial association before any rule fires.
4. Multi-frame person tracking to avoid duplicate violation counting.
5. Restricted-zone entry detection via polygon geometry.
6. Evidence capture (annotated frame + metadata) per violation.
7. Machine-readable (JSON/CSV) and human-readable (terminal) reporting.

# Assumptions

- Camera is roughly upright (head is at the top of a person's bounding box);
  the head-region heuristic assumes this.
- The PPE detector's accuracy is bounded by the size and diversity of the
  5000-image training dataset — it will not generalize perfectly to every
  camera angle, lighting condition, or helmet color/style.
- "Violation" here means "the pipeline's evidence suggests a violation" —
  it is not a legal or disciplinary determination.

# Limitations

- Helmet compliance only; vest and other PPE classes are out of scope for
  this submission (documented as future work).
- Tracker is a simple greedy IoU tracker — it can lose or swap IDs under
  heavy occlusion or fast motion; it is not DeepSORT-grade.
- Restricted-zone check is 2D image-space geometry, not real-world
  ground-plane calibration.
- No identity/biometric recognition of any kind is performed or intended.
