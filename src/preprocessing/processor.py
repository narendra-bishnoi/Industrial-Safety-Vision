"""Input loading, validation, and frame extraction.

This is the only module allowed to touch raw file I/O for media — every
other module receives already-decoded numpy frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger("safety_vision.preprocessing")


class InputError(ValueError):
    """Raised when an input file cannot be read or decoded."""


@dataclass
class Frame:
    """A single decoded frame plus its position in the source media."""

    image: np.ndarray
    index: int
    timestamp_sec: float


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"Image file not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise InputError(f"OpenCV could not decode image (corrupt or unsupported format): {path}")
    return image


def resize_max_dim(image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    """Downscale so the longer side is at most max_dim; upscaling is never applied."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


class VideoFrameReader:
    """Iterates decoded frames from a video file with index/timestamp metadata."""

    def __init__(self, path: str | Path, frame_stride: int = 1):
        self.path = Path(path)
        if not self.path.is_file():
            raise InputError(f"Video file not found: {self.path}")
        if frame_stride < 1:
            raise InputError("frame_stride must be >= 1")
        self.frame_stride = frame_stride
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise InputError(f"OpenCV could not open video (corrupt or unsupported codec): {self.path}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0:
            logger.warning("Video reports FPS<=0; timestamps will be reported as frame index.")

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            if index % self.frame_stride == 0:
                timestamp = index / self.fps if self.fps > 0 else float(index)
                yield Frame(image=image, index=index, timestamp_sec=timestamp)
            index += 1
        self._cap.release()

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()
