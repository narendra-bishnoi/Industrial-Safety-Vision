"""Object detection wrappers around Ultralytics YOLO.

Two separate models are used deliberately (see docs/architecture.md):

- ``PersonDetector`` uses the stock COCO-pretrained YOLOv8n weights, which
  Ultralytics ships and auto-downloads. It is general-purpose and was not
  trained by us — we only use its existing "person" class.
- ``PPEDetector`` uses a YOLOv8n checkpoint fine-tuned by us (see
  scripts/train_ppe_model.py) on the Kaggle "Safety Helmet Detection"
  dataset, giving it the "helmet" / "head" classes that COCO does not have.

Both share the same thin interface so the rule engine never has to know
which model produced a detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("safety_vision.detection")

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coordinates


class ModelLoadError(RuntimeError):
    """Raised when a YOLO checkpoint cannot be loaded."""


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_name: str


def _load_yolo(weights: str, device: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment issue, not a logic bug
        raise ModelLoadError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        model = YOLO(weights)
    except Exception as exc:  # ultralytics raises plain Exception on bad/missing weights
        raise ModelLoadError(f"Could not load YOLO weights '{weights}': {exc}") from exc
    model.to(device)
    return model


class _YoloDetectorBase:
    """Shared inference logic; subclasses fix which classes they keep."""

    def __init__(self, weights: str, device: str = "cpu", confidence: float = 0.4):
        self.weights = weights
        self.confidence = confidence
        self._model = _load_yolo(weights, device)

    def _run(self, image: np.ndarray, keep_classes: set[str] | None) -> list[Detection]:
        results = self._model.predict(
            image, conf=self.confidence, device=self._model.device, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_name = names[int(box.cls[0])]
                if keep_classes is not None and class_name not in keep_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                        class_name=class_name,
                    )
                )
        return detections


class PersonDetector(_YoloDetectorBase):
    """Localizes people using the COCO 'person' class only."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self._run(image, keep_classes={"person"})


class PPEDetector(_YoloDetectorBase):
    """Detects head-PPE state ('helmet' = compliant, 'head' = bare head)."""

    def __init__(self, weights: str, device: str = "cpu", confidence: float = 0.4):
        if not Path(weights).is_file():
            raise ModelLoadError(
                f"PPE model weights not found: {weights}\n"
                "Train them with scripts/train_ppe_model.py after placing the "
                "Kaggle Safety Helmet Detection dataset in data/raw/ "
                "(see data/README.md)."
            )
        super().__init__(weights, device=device, confidence=confidence)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self._run(image, keep_classes={"helmet", "head"})
