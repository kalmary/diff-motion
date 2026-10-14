from pathlib import Path

import diff_motion
import pytest


def test_run_is_exposed():
    assert callable(diff_motion.run)


def test_run_signature_exposes_runtime_configuration():
    signature = diff_motion.run.__text_signature__

    assert "headless=True" in signature
    assert "processing_type=None" in signature
    assert "camera_index=None" in signature
    assert "video_path=None" in signature
    assert "callback=None" in signature


def test_run_reports_missing_video_source(tmp_path):
    missing_video = tmp_path / "missing.mp4"

    with pytest.raises(RuntimeError, match="Failed to open video"):
        diff_motion.run(video_path=str(missing_video))


def test_run_delivers_first_frame_and_preserves_callback_exception():
    results = []
    error = ValueError("stop capture")

    def receive(result):
        results.append(result)
        raise error

    video = Path(__file__).resolve().parents[1] / "fixtures" / "gemini_generated_video_8813b1b0.mp4"
    with pytest.raises(ValueError) as caught:
        diff_motion.run(video_path=str(video), callback=receive)

    assert caught.value is error
    assert len(results) == 1
    mask = results[0]["mask"]
    assert mask["width"] > 0 and mask["height"] > 0
    assert isinstance(mask["data"], bytes)
    assert len(mask["data"]) == mask["width"] * mask["height"]
    assert set(mask["data"]) == {0}
    assert results[0]["objects"] == []


def test_run_rejects_non_callable_callback():
    with pytest.raises(TypeError, match="callback must be callable"):
        diff_motion.run(callback=42)


def test_run_delivers_consecutive_owned_results():
    results = []

    def receive(result):
        results.append(result)
        if len(results) == 3:
            raise StopIteration("enough frames")

    video = Path(__file__).resolve().parents[1] / "fixtures" / "gemini_generated_video_8813b1b0.mp4"
    with pytest.raises(StopIteration, match="enough frames"):
        diff_motion.run(video_path=str(video), callback=receive)

    assert len(results) == 3
    assert len({id(result) for result in results}) == 3
    for result in results:
        mask = result["mask"]
        assert len(mask["data"]) == mask["width"] * mask["height"]
        assert set(mask["data"]) <= {0, 1}
        for obj in result["objects"]:
            assert obj["class_id"] == 1
            assert obj["class_name"] == "UAV"
            bbox = obj["bbox"]
            assert 0 <= bbox["x"] < mask["width"]
            assert 0 <= bbox["y"] < mask["height"]
            assert bbox["width"] > 0 and bbox["height"] > 0
    assert set(results[0]["mask"]["data"]) == {0}
