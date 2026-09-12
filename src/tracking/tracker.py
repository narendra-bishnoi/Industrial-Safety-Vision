"""Greedy IoU-based multi-object tracker.

ponytail: greedy highest-IoU assignment, not the Hungarian algorithm — fine
for the handful of people typically in one frame of construction footage.
Upgrade to `scipy.optimize.linear_sum_assignment` if a scene needs globally
optimal assignment among many overlapping people.

This is our own small implementation of the standard "IoU tracker" concept
(cf. Bochinski et al., 2017, "High-Speed Tracking-by-Detection Without Using
Image Information") — conceptually related to SORT, not a copy of any
tracking library's code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from src.detection.detector import BBox


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    missed_frames: int = 0


@dataclass
class IouTracker:
    """Assigns stable IDs to person boxes across consecutive frames."""

    iou_threshold: float = 0.3
    max_age: int = 30
    total_tracks_created: int = 0
    _tracks: list[_Track] = field(default_factory=list)
    _next_id: itertools.count = field(default_factory=lambda: itertools.count(1))

    def update(self, detections: list[BBox]) -> list[tuple[int, BBox]]:
        """Call once per frame with the current frame's person boxes.

        Returns (track_id, bbox) pairs for every detection in this frame.
        """
        unmatched_detections = list(range(len(detections)))
        results: list[tuple[int, BBox]] = [None] * len(detections)  # type: ignore[list-item]

        # Greedily match each existing track to its best remaining detection.
        for track in self._tracks:
            if not unmatched_detections:
                break
            best_idx, best_iou = None, 0.0
            for idx in unmatched_detections:
                score = iou(track.bbox, detections[idx])
                if score > best_iou:
                    best_idx, best_iou = idx, score
            if best_idx is not None and best_iou >= self.iou_threshold:
                track.bbox = detections[best_idx]
                track.missed_frames = 0
                results[best_idx] = (track.track_id, detections[best_idx])
                unmatched_detections.remove(best_idx)

        # Age out tracks that found no match this frame.
        for track in self._tracks:
            if all(track.track_id != r[0] for r in results if r is not None):
                track.missed_frames += 1
        self._tracks = [t for t in self._tracks if t.missed_frames <= self.max_age]

        # Start new tracks for detections nothing matched.
        for idx in unmatched_detections:
            new_id = next(self._next_id)
            self._tracks.append(_Track(track_id=new_id, bbox=detections[idx]))
            self.total_tracks_created += 1
            results[idx] = (new_id, detections[idx])

        return results  # type: ignore[return-value]
