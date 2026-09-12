"""Turns raw detections into safety decisions.

This is the module that must be explainable in the viva: it is NOT
"if helmet not detected: violation" applied globally. It performs, per
person, a spatial association step before any rule fires:

    for each tracked person:
        head_region = top {HEAD_REGION_FRACTION} of their bounding box
        find PPE detections whose center lands inside that head_region
        -> if a 'helmet' detection is found there: COMPLIANT
        -> if only a 'head' (bare-head) detection is found there: VIOLATION
        -> if nothing is found there: VIOLATION, but marked low-confidence
           (UNVERIFIED) because absence of evidence is weaker than a
           positive 'head' detection — this distinction is the difference
           between "detection" and "inference" the assignment asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.config import HEAD_REGION_FRACTION
from src.detection.detector import BBox, Detection


class ViolationType(str, Enum):
    NO_HELMET = "NO_HELMET"  # a bare head was positively detected -> high confidence
    NO_HELMET_UNVERIFIED = "NO_HELMET_UNVERIFIED"  # no helmet AND no head found -> weaker inference
    RESTRICTED_ZONE_ENTRY = "RESTRICTED_ZONE_ENTRY"


@dataclass
class ViolationEvent:
    track_id: int
    person_bbox: BBox
    violation_type: ViolationType
    confidence: float
    detail: str = ""


def _head_region(person_bbox: BBox, fraction: float = HEAD_REGION_FRACTION) -> BBox:
    x1, y1, x2, y2 = person_bbox
    height = y2 - y1
    return (x1, y1, x2, y1 + height * fraction)


def _center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def evaluate_helmet_compliance(
    persons: list[tuple[int, BBox]],
    ppe_detections: list[Detection],
) -> list[ViolationEvent]:
    """Associates PPE detections with each tracked person and returns violations.

    Compliant persons produce no event — only violations are reported, which
    keeps evidence/reporting focused on what a safety officer needs to see.
    """
    events: list[ViolationEvent] = []
    for track_id, person_bbox in persons:
        region = _head_region(person_bbox)
        matches = [d for d in ppe_detections if _point_in_bbox(_center(d.bbox), region)]

        helmet_matches = [d for d in matches if d.class_name == "helmet"]
        head_matches = [d for d in matches if d.class_name == "head"]

        if helmet_matches:
            continue  # compliant

        if head_matches:
            best = max(head_matches, key=lambda d: d.confidence)
            events.append(
                ViolationEvent(
                    track_id=track_id,
                    person_bbox=person_bbox,
                    violation_type=ViolationType.NO_HELMET,
                    confidence=best.confidence,
                    detail="Bare head detected in head region.",
                )
            )
        else:
            events.append(
                ViolationEvent(
                    track_id=track_id,
                    person_bbox=person_bbox,
                    violation_type=ViolationType.NO_HELMET_UNVERIFIED,
                    confidence=0.0,
                    detail="No helmet or head detected in head region (occlusion or missed detection).",
                )
            )
    return events
