from pathlib import Path

import pytest

from src.config import AppConfig, ConfigError, ZoneConfig


def test_invalid_confidence_raises(tmp_path):
    sample = tmp_path / "sample.jpg"
    sample.write_bytes(b"fake")
    with pytest.raises(ConfigError):
        AppConfig(input_path=sample, output_dir=tmp_path / "out", mode="image", confidence=1.5)


def test_invalid_mode_raises(tmp_path):
    sample = tmp_path / "sample.jpg"
    sample.write_bytes(b"fake")
    with pytest.raises(ConfigError):
        AppConfig(input_path=sample, output_dir=tmp_path / "out", mode="audio")


def test_missing_input_path_raises(tmp_path):
    with pytest.raises(ConfigError):
        AppConfig(input_path=tmp_path / "missing.jpg", output_dir=tmp_path / "out", mode="image")


def test_mismatched_extension_for_mode_raises(tmp_path):
    sample = tmp_path / "sample.mp4"
    sample.write_bytes(b"fake")
    with pytest.raises(ConfigError):
        AppConfig(input_path=sample, output_dir=tmp_path / "out", mode="image")


def test_valid_config_creates_output_dir(tmp_path):
    sample = tmp_path / "sample.jpg"
    sample.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    AppConfig(input_path=sample, output_dir=out_dir, mode="image")
    assert out_dir.is_dir()


def test_zone_config_rejects_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        ZoneConfig.load(tmp_path / "no_such_zones.json")


def test_zone_config_rejects_polygon_with_too_few_points(tmp_path):
    zones_file = tmp_path / "zones.json"
    zones_file.write_text('{"zones": [{"name": "bad", "points": [[0,0],[1,1]]}]}', encoding="utf-8")
    with pytest.raises(ConfigError):
        ZoneConfig.load(zones_file)


def test_zone_config_loads_valid_polygon(tmp_path):
    zones_file = tmp_path / "zones.json"
    zones_file.write_text(
        '{"zones": [{"name": "crane_area", "points": [[0,0],[10,0],[10,10],[0,10]]}]}',
        encoding="utf-8",
    )
    zones = ZoneConfig.load(zones_file)
    assert len(zones) == 1
    assert zones[0].name == "crane_area"
