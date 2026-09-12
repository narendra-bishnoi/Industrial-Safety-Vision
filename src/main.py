"""Command-line entry point for the Industrial Safety Vision System.

Example:
    python -m src.main --input data/sample.jpg --mode image --output outputs/
    python -m src.main --input data/sample.mp4 --mode video --output outputs/ --confidence 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from src.config import AppConfig, ConfigError, DEFAULT_PERSON_MODEL, DEFAULT_PPE_MODEL, ZoneConfig
from src.detection.detector import ModelLoadError, PersonDetector, PPEDetector
from src.evidence.extractor import capture_evidence
from src.preprocessing.processor import InputError, VideoFrameReader, load_image, resize_max_dim
from src.reporting.reporter import RunStats, print_text_summary, write_csv_report, write_json_report
from src.safety.rule_engine import evaluate_helmet_compliance
from src.safety.zones import check_restricted_zones
from src.tracking.tracker import IouTracker
from src.utils.logger import setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Industrial Safety Vision System")
    parser.add_argument("--input", required=True, help="Path to an image or video file")
    parser.add_argument("--output", default="outputs", help="Directory for annotated media, evidence, and reports")
    parser.add_argument("--mode", choices=["image", "video"], help="Defaults to auto-detect from file extension")
    parser.add_argument("--confidence", type=float, default=0.4, help="Detection confidence threshold (0-1]")
    parser.add_argument("--person-model", default=DEFAULT_PERSON_MODEL)
    parser.add_argument("--ppe-model", default=DEFAULT_PPE_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--save-evidence", action="store_true", default=True)
    parser.add_argument("--no-save-evidence", dest="save_evidence", action="store_false")
    parser.add_argument("--zones", default=None, help="Path to a JSON file defining restricted zones")
    parser.add_argument("--report", default="json,text", help="Comma-separated: json,csv,text")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth video frame")
    return parser.parse_args(argv)


def _infer_mode(input_path: Path) -> str:
    from src.config import VALID_IMAGE_EXT, VALID_VIDEO_EXT

    suffix = input_path.suffix.lower()
    if suffix in VALID_IMAGE_EXT:
        return "image"
    if suffix in VALID_VIDEO_EXT:
        return "video"
    raise ConfigError(f"Cannot infer mode from extension '{suffix}'; pass --mode explicitly")


def build_config(args: argparse.Namespace) -> AppConfig:
    input_path = Path(args.input)
    mode = args.mode or _infer_mode(input_path)
    zones = ZoneConfig.load(args.zones) if args.zones else []
    return AppConfig(
        input_path=input_path,
        output_dir=Path(args.output),
        mode=mode,
        confidence=args.confidence,
        person_model=args.person_model,
        ppe_model=args.ppe_model,
        device=args.device,
        save_evidence=args.save_evidence,
        report_formats=tuple(f.strip() for f in args.report.split(",") if f.strip()),
        zones=zones,
    )


def _process_frame(image, person_detector, ppe_detector, tracker, config, stats, logger, frame_index, timestamp):
    persons = person_detector.detect(image)
    ppe_detections = ppe_detector.detect(image)

    person_boxes = [p.bbox for p in persons]
    tracked = tracker.update(person_boxes)
    stats.unique_persons_tracked = tracker.total_tracks_created

    events = evaluate_helmet_compliance(tracked, ppe_detections)
    events += check_restricted_zones(tracked, config.zones)

    annotated = image.copy()
    for track_id, bbox in tracked:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(annotated, f"ID {track_id}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    for event in events:
        if config.save_evidence:
            record = capture_evidence(
                image, event, frame_index, timestamp, config.output_dir / "evidence"
            )
            stats.evidence.append(record)
        logger.warning("Violation: track=%s type=%s detail=%s", event.track_id, event.violation_type.value, event.detail)

    return annotated


def run(config: AppConfig) -> RunStats:
    logger = setup_logger(config.output_dir / "logs")
    stats = RunStats(input_path=str(config.input_path))

    logger.info("Loading person detector: %s", config.person_model)
    person_detector = PersonDetector(config.person_model, device=config.device, confidence=config.confidence)
    logger.info("Loading PPE detector: %s", config.ppe_model)
    ppe_detector = PPEDetector(config.ppe_model, device=config.device, confidence=config.confidence)
    tracker = IouTracker(iou_threshold=config.iou_match_threshold, max_age=config.max_track_age)

    start = time.perf_counter()

    if config.mode == "image":
        image = resize_max_dim(load_image(config.input_path))
        annotated = _process_frame(image, person_detector, ppe_detector, tracker, config, stats, logger, 0, 0.0)
        out_path = config.output_dir / f"annotated_{config.input_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        stats.total_frames_processed = 1
        logger.info("Annotated image written to %s", out_path)
    else:
        reader = VideoFrameReader(config.input_path)
        out_path = config.output_dir / f"annotated_{config.input_path.stem}.mp4"
        writer = None
        for frame in reader:
            image = resize_max_dim(frame.image)
            annotated = _process_frame(
                image, person_detector, ppe_detector, tracker, config, stats, logger,
                frame.index, frame.timestamp_sec,
            )
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), reader.fps or 25.0, (w, h))
            writer.write(annotated)
            stats.total_frames_processed += 1
        if writer is not None:
            writer.release()
        logger.info("Annotated video written to %s", out_path)

    stats.inference_seconds = time.perf_counter() - start

    if "json" in config.report_formats:
        write_json_report(stats, config.output_dir)
    if "csv" in config.report_formats:
        write_csv_report(stats, config.output_dir)
    if "text" in config.report_formats:
        print_text_summary(stats)

    return stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = build_config(args)
        run(config)
        return 0
    except (ConfigError, InputError, ModelLoadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
