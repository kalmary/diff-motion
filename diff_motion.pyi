from collections.abc import Callable
from typing import Any, Literal

def run(
    headless: bool | None = None,
    processing_type: str | None = None,
    camera_index: int | None = None,
    video_path: str | None = None,
    callback: Callable[[dict[str, Any]], object] | None = None,
    *,
    config_path: str | None = None,
    mode: Literal["none", "info", "debug"] | None = None,
    output_mode: str | None = None,
    resolution: tuple[int, int] | None = None,
    input_resolution: tuple[int, int] | None = None,
    display_resolution: tuple[int, int] | None = None,
    fps: float | None = None,
    fb_max: float | None = None,
    ransac_h: float | None = None,
    tau_min: float | None = None,
    k_sigma: float | None = None,
    tau_epi: float | None = None,
    max_corners: int | None = None,
    quality_level: float | None = None,
    min_distance: float | None = None,
    min_inliers: int | None = None,
    parallax_threshold: float | None = None,
    flow_preset: str | None = None,
    finest_scale: int | None = None,
    morphology_open_size: int | None = None,
    morphology_close_size: int | None = None,
    min_component_area: int | None = None,
    search_score_threshold: float | None = None,
    tracking_score_threshold: float | None = None,
    confirmation_window: int | None = None,
    confirmation_hits: int | None = None,
    tracker_gate_scale: float | None = None,
    max_coast_frames: int | None = None,
    min_correlation: float | None = None,
) -> None:
    """Process video; max_coast_frames=0 disables propagated masks.

    min_correlation checks appearance for observations and coasting.
    Logging info includes reliability and errors; debug also includes warnings.
    """
    ...
