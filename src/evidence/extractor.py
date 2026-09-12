"""Captures an image + metadata record for every violation event."""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from src.safety.rule_engine import ViolationEvent

logger = logging.getLogger("safety_vision.evidence")


@dataclass
class EvidenceRecord:
    track_id: int
    frame_index: int
    timestamp_sec: float
    violation_type: str
    confidence: float
    detail: str
    evidence_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _draw_box(image: np.ndarray, bbox: tuple[float, float, float, float], label: str) -> np.ndarray:
    annotated = image.copy()
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(annotated, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return annotated


def capture_evidence(
    frame: np.ndarray,
    event: ViolationEvent,
    frame_index: int,
    timestamp_sec: float,
    output_dir: str | Path,
) -> EvidenceRecord:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"frame{frame_index:06d}_track{event.track_id}_{event.violation_type.value}.jpg"
    path = output_dir / filename

    annotated = _draw_box(frame, event.person_bbox, event.violation_type.value)
    ok = cv2.imwrite(str(path), annotated)
    if not ok:
        logger.error("Failed to write evidence image: %s", path)

    return EvidenceRecord(
        track_id=event.track_id,
        frame_index=frame_index,
        timestamp_sec=round(timestamp_sec, 3),
        violation_type=event.violation_type.value,
        confidence=round(event.confidence, 4),
        detail=event.detail,
        evidence_path=str(path),
    )
