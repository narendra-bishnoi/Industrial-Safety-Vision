import json

from src.evidence.extractor import EvidenceRecord
from src.reporting.reporter import RunStats, write_csv_report, write_json_report


def _sample_stats() -> RunStats:
    stats = RunStats(input_path="data/sample.mp4", total_frames_processed=10, unique_persons_tracked=2)
    stats.evidence = [
        EvidenceRecord(
            track_id=1,
            frame_index=5,
            timestamp_sec=1.2,
            violation_type="NO_HELMET",
            confidence=0.83,
            detail="Bare head detected in head region.",
            evidence_path="outputs/evidence/frame000005_track1_NO_HELMET.jpg",
        )
    ]
    return stats


def test_json_report_contains_expected_fields(tmp_path):
    stats = _sample_stats()
    path = write_json_report(stats, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["total_frames_processed"] == 10
    assert data["total_violations"] == 1
    assert data["violations_by_type"] == {"NO_HELMET": 1}
    assert data["evidence"][0]["track_id"] == 1


def test_csv_report_has_header_and_row(tmp_path):
    stats = _sample_stats()
    path = write_csv_report(stats, tmp_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert lines[0].startswith("track_id,")
    assert len(lines) == 2  # header + one violation


def test_json_report_with_no_violations_has_empty_counts(tmp_path):
    stats = RunStats(input_path="data/clean.mp4", total_frames_processed=5)
    path = write_json_report(stats, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["total_violations"] == 0
    assert data["violations_by_type"] == {}
