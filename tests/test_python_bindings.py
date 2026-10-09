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


def test_run_reports_missing_video_source(tmp_path):
    missing_video = tmp_path / "missing.mp4"

    with pytest.raises(RuntimeError, match="Failed to open video"):
        diff_motion.run(video_path=str(missing_video))
