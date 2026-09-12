"""Aggregates run statistics and evidence into machine- and human-readable reports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.evidence.extractor import EvidenceRecord


@dataclass
class RunStats:
    input_path: str
    total_frames_processed: int = 0
    unique_persons_tracked: int = 0
    evidence: list[EvidenceRecord] = field(default_factory=list)
    inference_seconds: float = 0.0

    def violation_counts(self) -> dict[str, int]:
        return dict(Counter(e.violation_type for e in self.evidence))

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_frames_processed": self.total_frames_processed,
            "unique_persons_tracked": self.unique_persons_tracked,
            "total_violations": len(self.evidence),
            "violations_by_type": self.violation_counts(),
            "inference_seconds": round(self.inference_seconds, 3),
            "fps": round(self.total_frames_processed / self.inference_seconds, 2)
            if self.inference_seconds > 0
            else None,
            "evidence": [e.to_dict() for e in self.evidence],
        }


def write_json_report(stats: RunStats, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "report.json"
    path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    return path


def write_csv_report(stats: RunStats, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "violations.csv"
    fieldnames = ["track_id", "frame_index", "timestamp_sec", "violation_type", "confidence", "detail", "evidence_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in stats.evidence:
            writer.writerow(e.to_dict())
    return path


def print_text_summary(stats: RunStats) -> str:
    lines = [
        "=" * 60,
        "INDUSTRIAL SAFETY VISION SYSTEM — RUN SUMMARY",
        "=" * 60,
        f"Input:                 {stats.input_path}",
        f"Frames processed:      {stats.total_frames_processed}",
        f"Unique persons seen:   {stats.unique_persons_tracked}",
        f"Total violations:      {len(stats.evidence)}",
    ]
    counts = stats.violation_counts()
    if counts:
        lines.append("Violations by type:")
        for vtype, count in sorted(counts.items()):
            lines.append(f"  - {vtype}: {count}")
    if stats.inference_seconds > 0:
        fps = stats.total_frames_processed / stats.inference_seconds
        lines.append(f"Inference time:        {stats.inference_seconds:.2f}s ({fps:.2f} FPS)")
    lines.append("=" * 60)
    text = "\n".join(lines)
    print(text)
    return text
