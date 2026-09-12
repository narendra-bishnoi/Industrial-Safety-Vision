"""Restricted-zone checking via point-in-polygon.

A person is "in a zone" if the bottom-center point of their bounding box
(a standard proxy for foot position in image-space) lies inside the zone
polygon. This is plain 2D geometry, not a learned capability — documented
here so it is never mistaken for an AI claim.
"""

from __future__ import annotations

from src.config import ZoneConfig
from src.detection.detector import BBox
from src.safety.rule_engine import ViolationEvent, ViolationType


def _foot_point(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, y2


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting algorithm (even-odd rule)."""
    x, y = point
    inside = False
    n = len(polygon)
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def check_restricted_zones(
    persons: list[tuple[int, BBox]],
    zones: list[ZoneConfig],
) -> list[ViolationEvent]:
    events: list[ViolationEvent] = []
    for track_id, bbox in persons:
        point = _foot_point(bbox)
        for zone in zones:
            if _point_in_polygon(point, zone.points):
                events.append(
                    ViolationEvent(
                        track_id=track_id,
                        person_bbox=bbox,
                        violation_type=ViolationType.RESTRICTED_ZONE_ENTRY,
                        confidence=1.0,  # geometric fact, not a model confidence
                        detail=f"Entered restricted zone '{zone.name}'.",
                    )
                )
    return events
