from src.tracking.tracker import IouTracker, iou


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_same_person_keeps_id_across_frames():
    tracker = IouTracker(iou_threshold=0.3)
    frame1 = [(10.0, 10.0, 50.0, 100.0)]
    frame2 = [(12.0, 11.0, 52.0, 101.0)]  # small movement, high IoU with frame1

    result1 = tracker.update(frame1)
    result2 = tracker.update(frame2)

    assert result1[0][0] == result2[0][0]


def test_new_person_gets_new_id():
    tracker = IouTracker(iou_threshold=0.3)
    tracker.update([(10.0, 10.0, 50.0, 100.0)])
    result = tracker.update([(10.0, 10.0, 50.0, 100.0), (500.0, 500.0, 540.0, 600.0)])

    ids = {r[0] for r in result}
    assert len(ids) == 2


def test_track_expires_after_max_age():
    tracker = IouTracker(iou_threshold=0.3, max_age=1)
    tracker.update([(10.0, 10.0, 50.0, 100.0)])
    tracker.update([])  # miss 1 of 1 tolerated -> track survives
    tracker.update([])  # miss 2 -> exceeds max_age=1, track pruned
    tracker.update([(10.0, 10.0, 50.0, 100.0)])  # reappears at same spot -> new track
    assert tracker.total_tracks_created == 2
