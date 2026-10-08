import pytest
import diff_motion

def test_run_headless(capsys):
    # This should call the rust code and we can capture stdout to verify config
    diff_motion.run(headless=True)
    captured = capsys.readouterr()
    assert "Headless" in captured.out

def test_run_display(capsys):
    diff_motion.run(headless=False)
    captured = capsys.readouterr()
    assert "Display" in captured.out

def test_run_video_path(capsys):
    diff_motion.run(video_path="test_video.mp4")
    captured = capsys.readouterr()
    assert "test_video.mp4" in captured.out
