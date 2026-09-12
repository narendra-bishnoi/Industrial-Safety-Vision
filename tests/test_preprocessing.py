import numpy as np
import pytest

from src.preprocessing.processor import InputError, VideoFrameReader, load_image, resize_max_dim


def test_load_image_missing_file_raises():
    with pytest.raises(InputError):
        load_image("data/does_not_exist.jpg")


def test_load_image_corrupt_file_raises(tmp_path):
    bad_file = tmp_path / "not_an_image.jpg"
    bad_file.write_text("this is not image data")
    with pytest.raises(InputError):
        load_image(bad_file)


def test_resize_max_dim_shrinks_large_image():
    image = np.zeros((2000, 1000, 3), dtype=np.uint8)
    resized = resize_max_dim(image, max_dim=1000)
    assert max(resized.shape[:2]) == 1000


def test_resize_max_dim_leaves_small_image_untouched():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_max_dim(image, max_dim=1280)
    assert resized.shape == image.shape


def test_video_reader_missing_file_raises():
    with pytest.raises(InputError):
        VideoFrameReader("data/does_not_exist.mp4")


def test_video_reader_rejects_bad_stride(tmp_path):
    fake_video = tmp_path / "clip.mp4"
    fake_video.write_bytes(b"\x00")
    with pytest.raises(InputError):
        VideoFrameReader(fake_video, frame_stride=0)
