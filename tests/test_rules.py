from src.detection.detector import Detection
from src.safety.rule_engine import ViolationType, evaluate_helmet_compliance
from src.safety.zones import check_restricted_zones
from src.config import ZoneConfig

PERSON_BBOX = (100.0, 100.0, 200.0, 400.0)  # head region ~= y in [100, 190]


def test_helmet_present_in_head_region_is_compliant():
    persons = [(1, PERSON_BBOX)]
    ppe = [Detection(bbox=(120, 110, 180, 160), confidence=0.9, class_name="helmet")]
    events = evaluate_helmet_compliance(persons, ppe)
    assert events == []


def test_bare_head_in_head_region_is_no_helmet_violation():
    persons = [(1, PERSON_BBOX)]
    ppe = [Detection(bbox=(120, 110, 180, 160), confidence=0.8, class_name="head")]
    events = evaluate_helmet_compliance(persons, ppe)
    assert len(events) == 1
    assert events[0].violation_type == ViolationType.NO_HELMET
    assert events[0].track_id == 1


def test_no_detection_in_head_region_is_unverified_violation():
    persons = [(1, PERSON_BBOX)]
    events = evaluate_helmet_compliance(persons, ppe_detections=[])
    assert len(events) == 1
    assert events[0].violation_type == ViolationType.NO_HELMET_UNVERIFIED


def test_ppe_detection_outside_head_region_is_ignored():
    persons = [(1, PERSON_BBOX)]
    # A helmet-classed box centered near the person's feet should not count.
    ppe = [Detection(bbox=(120, 380, 180, 399), confidence=0.9, class_name="helmet")]
    events = evaluate_helmet_compliance(persons, ppe)
    assert len(events) == 1
    assert events[0].violation_type == ViolationType.NO_HELMET_UNVERIFIED


def test_multiple_people_evaluated_independently():
    persons = [(1, PERSON_BBOX), (2, (300.0, 100.0, 400.0, 400.0))]
    ppe = [Detection(bbox=(120, 110, 180, 160), confidence=0.9, class_name="helmet")]
    events = evaluate_helmet_compliance(persons, ppe)
    assert len(events) == 1
    assert events[0].track_id == 2


def test_restricted_zone_entry_detected():
    zone = ZoneConfig(name="danger_area", points=[(0, 0), (500, 0), (500, 500), (0, 500)])
    persons = [(1, PERSON_BBOX)]  # foot point = (150, 400), inside the square
    events = check_restricted_zones(persons, [zone])
    assert len(events) == 1
    assert events[0].violation_type.value == "RESTRICTED_ZONE_ENTRY"


def test_restricted_zone_not_triggered_outside_polygon():
    zone = ZoneConfig(name="danger_area", points=[(1000, 1000), (1100, 1000), (1100, 1100), (1000, 1100)])
    persons = [(1, PERSON_BBOX)]
    events = check_restricted_zones(persons, [zone])
    assert events == []
