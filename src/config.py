"""Central configuration for the Industrial Safety Vision System.

Keeping every tunable value in one dataclass (rather than scattering
constants through the pipeline) is what makes the CLI, the tests, and the
training script agree on the same defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """Raised when a configuration value is invalid or a required file is missing."""


DEFAULT_PERSON_MODEL = "yolov8n.pt"  # COCO-pretrained, auto-downloaded by ultralytics
DEFAULT_PPE_MODEL = "models/ppe_helmet_best.pt"  # produced by scripts/train_ppe_model.py
HEAD_REGION_FRACTION = 0.30  # top fraction of a person's bbox treated as the "head region"
VALID_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ZoneConfig:
    """A restricted zone as a polygon in pixel coordinates."""

    name: str
    points: list[tuple[float, float]]

    @staticmethod
    def load(path: str | Path) -> list["ZoneConfig"]:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Zone file not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Zone file is not valid JSON: {path}") from exc
        zones = []
        for entry in raw.get("zones", []):
            points = [tuple(p) for p in entry.get("points", [])]
            if len(points) < 3:
                raise ConfigError(f"Zone '{entry.get('name')}' needs at least 3 points")
            zones.append(ZoneConfig(name=entry.get("name", "zone"), points=points))
        return zones


@dataclass
class AppConfig:
    """Runtime configuration assembled from CLI arguments."""

    input_path: Path
    output_dir: Path
    mode: str  # "image" | "video"
    confidence: float = 0.4
    person_model: str = DEFAULT_PERSON_MODEL
    ppe_model: str = DEFAULT_PPE_MODEL
    device: str = "cpu"
    save_evidence: bool = True
    report_formats: tuple[str, ...] = ("json", "text")
    zones: list[ZoneConfig] = field(default_factory=list)
    max_track_age: int = 30
    iou_match_threshold: float = 0.3

    def __post_init__(self) -> None:
        if not (0.0 < self.confidence <= 1.0):
            raise ConfigError(f"confidence must be in (0, 1], got {self.confidence}")
        if self.mode not in ("image", "video"):
            raise ConfigError(f"mode must be 'image' or 'video', got {self.mode!r}")
        if not self.input_path.exists():
            raise ConfigError(f"Input path does not exist: {self.input_path}")
        suffix = self.input_path.suffix.lower()
        if self.mode == "image" and suffix not in VALID_IMAGE_EXT:
            raise ConfigError(f"Unsupported image extension: {suffix}")
        if self.mode == "video" and suffix not in VALID_VIDEO_EXT:
            raise ConfigError(f"Unsupported video extension: {suffix}")
        if self.max_track_age <= 0:
            raise ConfigError("max_track_age must be positive")
        if not (0.0 < self.iou_match_threshold <= 1.0):
            raise ConfigError("iou_match_threshold must be in (0, 1]")
        self.output_dir.mkdir(parents=True, exist_ok=True)
