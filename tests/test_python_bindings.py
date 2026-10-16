from pathlib import Path
from typing import Any, cast

import diff_motion
import pytest


def test_run_is_exposed():
    assert callable(diff_motion.run)


def test_run_signature_exposes_runtime_configuration():
    signature = getattr(diff_motion.run, "__text_signature__")

    assert "headless=None" in signature
    assert "processing_type=None" in signature
    assert "camera_index=None" in signature
    assert "video_path=None" in signature
    assert "callback=None, *, config_path=None" in signature
    for name in [
        "mode",
        "output_mode",
        "resolution",
        "input_resolution",
        "display_resolution",
        "fps",
        "fb_max",
        "ransac_h",
        "tau_min",
        "k_sigma",
        "tau_epi",
        "max_corners",
        "quality_level",
        "min_distance",
        "min_inliers",
        "parallax_threshold",
        "flow_preset",
        "finest_scale",
        "morphology_open_size",
        "morphology_close_size",
        "min_component_area",
        "search_score_threshold",
        "tracking_score_threshold",
        "confirmation_window",
        "confirmation_hits",
        "tracker_gate_scale",
        "max_coast_frames",
        "min_correlation",
    ]:
        assert f"{name}=None" in signature


def test_run_uses_shared_default_processing_resolution():
    results = []

    def receive(result):
        results.append(result)
        raise StopIteration("first frame")

    video = Path(__file__).resolve().parents[1] / "fixtures" / "gemini_generated_video_8813b1b0.mp4"
    default_config = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    for config_path in [None, str(default_config)]:
        with pytest.raises(StopIteration, match="first frame"):
            diff_motion.run(video_path=str(video), config_path=config_path, callback=receive)

    dimensions = [(result["mask"]["width"], result["mask"]["height"]) for result in results]
    assert dimensions[0] == dimensions[1]


def test_yaml_configuration_is_loaded_and_keyword_arguments_override_it(tmp_path):
    results = []

    def receive(result):
        results.append(result)
        raise StopIteration("first frame")

    video = Path(__file__).resolve().parents[1] / "fixtures" / "gemini_generated_video_8813b1b0.mp4"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
video_path: {video}
algorithm_config:
  input_resolution:
    width: 160
    height: 90
  tau_min: -1
"""
    )

    with pytest.raises(StopIteration, match="first frame"):
        diff_motion.run(
            config_path=str(config),
            input_resolution=(320, 180),
            tau_min=0.9,
            callback=receive,
        )

    mask = results[0]["mask"]
    assert (mask["width"], mask["height"]) == (320, 180)


def test_run_rejects_unknown_yaml_keys_before_opening_source(tmp_path, capfd):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
video_path: missing.mp4
unknown: true
"""
    )

    with pytest.raises(ValueError, match="unknown"):
        diff_motion.run(config_path=str(config))

    assert capfd.readouterr().err.count("[ERROR]") == 1


def test_run_reports_missing_configuration_file(tmp_path):
    missing_config = tmp_path / "missing.yaml"

    with pytest.raises(RuntimeError, match="Failed to read config file"):
        diff_motion.run(config_path=str(missing_config))


def test_run_rejects_invalid_algorithm_setting_before_opening_source(tmp_path):
    with pytest.raises(ValueError, match="min_inliers"):
        diff_motion.run(
            video_path=str(tmp_path / "missing.mp4"),
            min_inliers=0,
        )


def test_run_accepts_fractional_fps_before_opening_source(tmp_path):
    with pytest.raises(RuntimeError, match="Failed to open video"):
        diff_motion.run(video_path=str(tmp_path / "missing.mp4"), fps=29.97)


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_run_rejects_invalid_fps_before_opening_source(tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        diff_motion.run(video_path=str(tmp_path / "missing.mp4"), fps=fps)


def test_run_rejects_conflicting_output_mode_arguments():
    with pytest.raises(ValueError, match="headless.*output_mode"):
        diff_motion.run(headless=True, output_mode="display")


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("resolution", (0, 480)),
        ("resolution", (640, -1)),
        ("resolution", (2**31, 480)),
        ("resolution", (10**100, 480)),
        ("input_resolution", (0, 240)),
        ("input_resolution", (1, 1)),
        ("input_resolution", (320, 2**31)),
        ("input_resolution", (320, -(10**100))),
        ("display_resolution", (0, 480)),
        ("display_resolution", (640, -1)),
    ],
)
def test_run_rejects_invalid_resolution_before_opening_source(tmp_path, argument, value):
    with pytest.raises(ValueError, match="resolution"):
        diff_motion.run(video_path=str(tmp_path / "missing.mp4"), **{argument: value})


def test_run_reports_missing_video_source(tmp_path, capfd):
    missing_video = tmp_path / "missing.mp4"

    with pytest.raises(RuntimeError, match="Failed to open video"):
        diff_motion.run(video_path=str(missing_video))

    assert capfd.readouterr().err.count("[ERROR]") == 1


def test_run_rejects_invalid_logging_mode(capfd):
    with pytest.raises(ValueError, match="logging mode"):
        diff_motion.run(mode=cast(Any, "verbose"))

    assert capfd.readouterr().err.count("[ERROR]") == 1


def test_none_logging_mode_suppresses_errors(tmp_path, capfd):
    with pytest.raises(RuntimeError, match="Failed to open video"):
        diff_motion.run(video_path=str(tmp_path / "missing.mp4"), mode="none")

    console = capfd.readouterr().err
    assert not any(f"[{level}]" in console for level in ("INFO", "WARN", "ERROR", "DEBUG"))


def test_none_logging_mode_suppresses_preflight_errors(tmp_path, capfd):
    missing_config = tmp_path / "missing.yaml"

    with pytest.raises(RuntimeError, match="Failed to read config file"):
        diff_motion.run(config_path=str(missing_config), mode="none")
    assert capfd.readouterr().err == ""

    with pytest.raises(TypeError, match="callback must be callable"):
        diff_motion.run(callback=cast(Any, 42), mode="none")
    assert capfd.readouterr().err == ""

    with pytest.raises(ValueError, match="resolution"):
        diff_motion.run(resolution=(0, 480), mode="none")
    assert capfd.readouterr().err == ""

    with pytest.raises(ValueError, match="fps"):
        diff_motion.run(fps=0.0, mode="none")
    assert capfd.readouterr().err == ""


def test_logging_mode_is_inherited_from_yaml(tmp_path, capfd):
    config = tmp_path / "config.yaml"
    config.write_text(
        f"mode: none\nvideo_path: {tmp_path / 'missing.mp4'}\n",
    )

    with pytest.raises(RuntimeError, match="Failed to open video"):
        diff_motion.run(config_path=str(config))

    console = capfd.readouterr().err
    assert not any(f"[{level}]" in console for level in ("INFO", "WARN", "ERROR", "DEBUG"))


def test_logging_mode_does_not_leak_between_runs(tmp_path, capfd):
    missing_video = tmp_path / "missing.mp4"

    with pytest.raises(RuntimeError, match="Failed to open video"):
        diff_motion.run(video_path=str(missing_video), mode="none")
    console = capfd.readouterr().err
    assert not any(f"[{level}]" in console for level in ("INFO", "WARN", "ERROR", "DEBUG"))

    with pytest.raises(RuntimeError, match="Failed to open video"):
        diff_motion.run(video_path=str(missing_video), mode="info")
    assert capfd.readouterr().err.count("[ERROR]") == 1


def test_zero_coast_frames_is_accepted():
    video = Path(__file__).resolve().parents[1] / "fixtures" / "gemini_generated_video_8813b1b0.mp4"

    def receive(_result):
        raise StopIteration("first frame")

    with pytest.raises(StopIteration, match="first frame"):
        diff_motion.run(video_path=str(video), max_coast_frames=0, mode="none", callback=receive)


@pytest.mark.parametrize("mode", ["info", "debug"])
def test_unreliable_motion_summary_uses_info_level(mode, capfd):
    video = Path(__file__).resolve().parents[1] / "fixtures" / "gemini_generated_video_8813b1b0.mp4"
    received = 0

    def receive(_result):
        nonlocal received
        received += 1
        if received == 2:
            raise StopIteration("two frames")

    with pytest.raises(StopIteration, match="two frames"):
        diff_motion.run(video_path=str(video), quality_level=1.0, mode=mode, callback=receive)

    console = capfd.readouterr().err
    summary = next(line for line in console.splitlines() if "Stream stopped:" in line)
    assert summary.startswith("[INFO]")
    assert "ego_motion_unreliable=100.00%" in summary
    if mode == "info":
        assert "[WARN]" not in console


def test_debug_logging_reports_processed_frame(capfd):
    video = Path(__file__).resolve().parents[1] / "fixtures" / "gemini_generated_video_8813b1b0.mp4"

    def stop_after_first_frame(_result):
        raise StopIteration("first frame")

    with pytest.raises(StopIteration, match="first frame"):
        diff_motion.run(video_path=str(video), callback=stop_after_first_frame, mode="debug")

    assert "[DEBUG]" in capfd.readouterr().err


def test_run_delivers_first_frame_and_preserves_callback_exception(capfd):
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
    console = capfd.readouterr().err
    assert console.count("[ERROR]") == 1
    assert "Stream stopped: fps=" in console
    assert "frames=" not in console
    assert "mask_0=100.00% mask_1=0.00%" in console


def test_run_rejects_non_callable_callback():
    with pytest.raises(TypeError, match="callback must be callable"):
        diff_motion.run(callback=cast(Any, 42))


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


def test_run_uses_requested_algorithm_input_resolution():
    results = []

    def receive(result):
        results.append(result)
        if len(results) == 3:
            raise StopIteration("enough frames")

    video = Path(__file__).resolve().parents[1] / "fixtures" / "gemini_generated_video_8813b1b0.mp4"
    with pytest.raises(StopIteration, match="enough frames"):
        diff_motion.run(
            video_path=str(video),
            callback=receive,
            input_resolution=(320, 180),
        )

    assert len(results) == 3
    for result in results:
        mask = result["mask"]
        assert (mask["width"], mask["height"]) == (320, 180)
        assert len(mask["data"]) == 320 * 180
        assert set(mask["data"]) <= {0, 1}
