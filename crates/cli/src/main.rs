use clap::{builder::TypedValueParser, Parser};
use diff_motion_core::{
    config::{AppConfig, FlowPreset, Mode, OutputMode, Resolution},
    logging::Logger,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn parse_camera_index(value: &str) -> Result<usize, String> {
    let index = value
        .parse::<usize>()
        .map_err(|_| format!("Camera index must be a non-negative integer: {value}"))?;
    i32::try_from(index)
        .map(|_| index)
        .map_err(|_| format!("Camera index {value} exceeds the OpenCV range"))
}

fn parse_fps(value: &str) -> Result<f64, String> {
    let fps = value
        .parse::<f64>()
        .map_err(|_| format!("FPS must be a positive finite number: {value}"))?;
    if fps.is_finite() && fps > 0.0 {
        Ok(fps)
    } else {
        Err(format!("FPS must be a positive finite number: {value}"))
    }
}

fn parse_resolution(value: &str) -> Result<Resolution, String> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| format!("Resolution must use WIDTHxHEIGHT: {value}"))?;
    let resolution = Resolution {
        width: width
            .parse()
            .map_err(|_| format!("Resolution width must be a positive integer: {value}"))?,
        height: height
            .parse()
            .map_err(|_| format!("Resolution height must be a positive integer: {value}"))?,
    };
    resolution
        .validate()
        .map_err(|error| format!("Invalid resolution {value}: {error}"))?;
    Ok(resolution)
}

fn parse_output_mode(value: &str) -> Result<OutputMode, String> {
    value.parse()
}

fn parse_flow_preset(value: &str) -> Result<FlowPreset, String> {
    value.parse()
}

fn mode_parser() -> impl TypedValueParser<Value = Mode> {
    clap::builder::PossibleValuesParser::new(["none", "info", "debug"])
        .map(|value| value.parse().expect("possible logging mode must parse"))
}

fn input_resolution_help() -> String {
    let resolution = AppConfig::default()
        .algorithm_config
        .resolved_input_resolution();
    format!(
        "Core processing and semantic mask resolution (default YAML: {}x{})\n\n\
         Main speed/detail control: larger images cost more work but preserve smaller \
         targets. Pixel thresholds and component areas use these dimensions. Preserve \
         the capture aspect ratio to avoid stretching; retune thresholds after resizing.",
        resolution.width, resolution.height
    )
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about,
    long_about = None,
    after_help = "Omitted options use values from the selected YAML configuration, except that a display window opens only when --display or --output-mode display is passed explicitly."
)]
pub struct Cli {
    /// Path to YAML configuration file
    ///
    /// Partial files inherit shipped defaults; explicit CLI values override the file.
    /// A Display value in YAML does not open a window without an explicit display CLI option.
    #[arg(
        short = 'c',
        long,
        visible_alias = "config",
        default_value = "config/default.yaml"
    )]
    pub config_path: PathBuf,

    /// Console logging level (default YAML: info)
    ///
    /// None suppresses application logs. Info reports startup, errors, FPS,
    /// mask pixel percentages and motion reliability. Debug also prints warnings,
    /// the full configuration and per-frame diagnostics.
    #[arg(long, value_parser = mode_parser(), help_heading = "Logging")]
    pub mode: Option<Mode>,

    /// Processing algorithm
    ///
    /// Only motion is implemented. Segmentation uses deterministic optical flow.
    #[arg(long, help_heading = "Algorithm")]
    pub processing_type: Option<String>,

    /// Display the camera image with a semantic mask overlay (q or Esc to quit)
    ///
    /// Explicitly opens the window. Shortcut for --output-mode display. Rendering adds
    /// work; without either display option the program runs headlessly.
    #[arg(long, conflicts_with = "output_mode", help_heading = "Output")]
    pub display: bool,

    /// Output mode: headless, display, or json-camera
    ///
    /// Headless still delivers callback results. Display adds an overlay and rendering
    /// cost and must be selected explicitly on the CLI. Json-camera is reserved and
    /// currently rejected.
    #[arg(long, value_parser = parse_output_mode, help_heading = "Output")]
    pub output_mode: Option<OutputMode>,

    /// Camera index
    ///
    /// Selects a backend-dependent camera device; ignored when a video file is supplied.
    #[arg(long, value_parser = parse_camera_index, help_heading = "Input")]
    pub camera_index: Option<usize>,

    /// Requested camera capture rate in frames per second (default YAML: 30)
    ///
    /// Passed to the camera backend, which may select a different supported rate.
    /// Ignored when a video file is supplied; measured processing FPS may be lower.
    #[arg(long, value_parser = parse_fps, help_heading = "Input")]
    pub fps: Option<f64>,

    /// Path to a video file for testing
    ///
    /// Overrides the camera source. Camera capture resolution does not resize file input.
    #[arg(long, help_heading = "Input")]
    pub video_path: Option<String>,

    /// Camera capture resolution (default YAML: camera-selected mode)
    ///
    /// Lower sizes can reduce capture/transfer work but lose source detail. The camera
    /// must support the requested size. Independent of processing and preview sizes.
    #[arg(long, value_name = "WIDTHxHEIGHT", value_parser = parse_resolution, help_heading = "Input")]
    pub resolution: Option<Resolution>,

    /// Display window resolution
    ///
    /// Changes preview size and display work only; masks and processing stay unchanged.
    /// The shipped default matches the captured frame.
    #[arg(long, value_name = "WIDTHxHEIGHT", value_parser = parse_resolution, help_heading = "Output")]
    pub display_resolution: Option<Resolution>,

    #[arg(
        long,
        value_name = "WIDTHxHEIGHT",
        value_parser = parse_resolution,
        help = input_resolution_help(),
        help_heading = "Algorithm"
    )]
    pub input_resolution: Option<Resolution>,

    /// Maximum forward-backward optical-flow error in pixels
    ///
    /// Lower rejects unreliable feature tracks more strictly, but may leave too few
    /// tracks for camera-motion estimation. Uses processing-image pixels.
    #[arg(long, value_name = "PIXELS", help_heading = "Algorithm")]
    pub fb_max: Option<f32>,

    /// RANSAC homography reprojection threshold in pixels
    ///
    /// Higher accepts noisier feature matches, potentially absorbing target motion
    /// into the background estimate. Uses processing-image pixels.
    #[arg(long, value_name = "PIXELS", help_heading = "Algorithm")]
    pub ransac_h: Option<f64>,

    /// Minimum adaptive motion threshold in pixels
    ///
    /// High-impact sensitivity control: lower detects weaker motion but more noise.
    /// Measured in processing-image pixels per frame; k-sigma can raise the threshold.
    #[arg(long, value_name = "PIXELS", help_heading = "Algorithm")]
    pub tau_min: Option<f32>,

    /// Robust-noise scale multiplier
    ///
    /// Adaptive threshold = max(tau-min, median + k-sigma * robust noise estimate).
    /// Higher suppresses background noise but can hide weak target motion.
    #[arg(long, help_heading = "Algorithm")]
    pub k_sigma: Option<f32>,

    /// Epipolar residual threshold in pixels
    ///
    /// Normalizes the epipolar score when a parallax model is available. Higher
    /// suppresses more residuals; has no effect without that model.
    #[arg(long, value_name = "PIXELS", help_heading = "Algorithm")]
    pub tau_epi: Option<f32>,

    /// Maximum number of detected feature corners
    ///
    /// More features can improve background coverage but increase sparse-flow and
    /// model-fitting work. At least 8; must not be smaller than min-inliers.
    #[arg(long, help_heading = "Algorithm")]
    pub max_corners: Option<i32>,

    /// Minimum feature-corner quality ratio
    ///
    /// Relative to the strongest corner, in (0, 1]. Higher selects stronger/fewer
    /// features; too high can prevent camera-motion estimation in low-texture scenes.
    #[arg(long, help_heading = "Algorithm")]
    pub quality_level: Option<f64>,

    /// Minimum distance between feature corners in pixels
    ///
    /// Higher spreads features out but reduces their count. Uses processing-image
    /// pixels, so its effect changes with input-resolution.
    #[arg(long, value_name = "PIXELS", help_heading = "Algorithm")]
    pub min_distance: Option<f64>,

    /// Minimum homography inlier count
    ///
    /// Required reliable matches, from 8 to max-corners. Higher demands stronger
    /// camera-motion evidence; insufficient inliers prevent new detections.
    #[arg(long, help_heading = "Algorithm")]
    pub min_inliers: Option<usize>,

    /// Minimum parallax fraction used for motion separation
    ///
    /// Attempts an epipolar model when the homography outlier fraction exceeds this
    /// value in [0, 1]. Lower activates parallax handling and its fitting work more often.
    #[arg(long, help_heading = "Algorithm")]
    pub parallax_threshold: Option<f64>,

    /// Dense optical-flow speed preset: ultrafast, fast, or medium
    ///
    /// Trades computation for flow refinement. Finest-scale separately overrides
    /// the preset's finest pyramid level; benchmark presets on representative footage.
    #[arg(long, value_parser = parse_flow_preset, help_heading = "Algorithm")]
    pub flow_preset: Option<FlowPreset>,

    /// Finest pyramid scale used for dense optical flow
    ///
    /// High-impact speed/detail control: 0 uses full resolution, 1 half, 2 quarter.
    /// Coarser levels can miss small targets. Maximum depends on input-resolution.
    #[arg(long, help_heading = "Algorithm")]
    pub finest_scale: Option<i32>,

    /// Morphological opening kernel size in pixels
    ///
    /// Positive odd width; 1 disables opening. Larger removes isolated noise but
    /// can erase small or thin targets. Uses processing-image pixels.
    #[arg(long, value_name = "PIXELS", help_heading = "Algorithm")]
    pub morphology_open_size: Option<i32>,

    /// Morphological closing kernel size in pixels
    ///
    /// Positive odd width; 1 disables closing. Larger fills gaps but may merge
    /// nearby foreground regions. Uses processing-image pixels.
    #[arg(long, value_name = "PIXELS", help_heading = "Algorithm")]
    pub morphology_close_size: Option<i32>,

    /// Minimum connected-component area in pixels
    ///
    /// Candidate area in processing pixels squared. Higher rejects small blobs,
    /// including small or distant targets; retune after changing input-resolution.
    #[arg(long, value_name = "PIXEL_COUNT", help_heading = "Algorithm")]
    pub min_component_area: Option<i32>,

    /// Minimum score for finding a new target
    ///
    /// Dimensionless per-pixel threshold outside a confirmed track's gate. Higher
    /// requires stronger motion evidence. Scores are not probabilities.
    #[arg(long, help_heading = "Algorithm")]
    pub search_score_threshold: Option<f32>,

    /// Minimum score while tracking a target
    ///
    /// Per-pixel threshold inside the tracking gate; must not exceed the search
    /// threshold. Lower retains weaker motion but admits more nearby noise.
    #[arg(long, help_heading = "Algorithm")]
    pub tracking_score_threshold: Option<f32>,

    /// Number of frames in the target-confirmation window
    ///
    /// Larger windows let intermittent detections accumulate but also retain older
    /// candidate history. Tune together with confirmation-hits; measured in frames.
    #[arg(long, value_name = "FRAMES", help_heading = "Algorithm")]
    pub confirmation_window: Option<usize>,

    /// Required detections in the target-confirmation window
    ///
    /// Higher reduces accidental acquisition but needs more frames to confirm.
    /// Must be positive and no greater than confirmation-window.
    #[arg(long, help_heading = "Algorithm")]
    pub confirmation_hits: Option<usize>,

    /// Scale applied to the tracker association gate
    ///
    /// Width/height multiplier around the predicted box, at least 1. Larger tolerates
    /// prediction error but admits competing motion. Camera-motion estimation excludes
    /// only the tracked box, not the expanded association gate.
    #[arg(long, help_heading = "Algorithm")]
    pub tracker_gate_scale: Option<f32>,

    /// Maximum frames to keep a target without a match
    ///
    /// Maximum consecutive template-propagated frames without a motion match.
    /// Zero disables mask propagation; larger values bridge longer pauses but risk
    /// retaining a stale target.
    #[arg(long, value_name = "FRAMES", help_heading = "Algorithm")]
    pub max_coast_frames: Option<u8>,

    /// Minimum tracker correlation score
    ///
    /// Template correlation in [-1, 1] for motion observations and coasting. Higher
    /// rejects appearance changes sooner; lower risks following an unrelated patch.
    #[arg(long, help_heading = "Algorithm")]
    pub min_correlation: Option<f64>,
}

fn parse_config(content: &str) -> Result<AppConfig, String> {
    AppConfig::from_yaml(content).map_err(|error| format!("Failed to parse config file: {error}"))
}

fn read_config(path: &Path) -> Result<AppConfig, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read config file {}: {error}", path.display()))?;
    parse_config(&content)
}

fn apply_overrides(mut config: AppConfig, cli: &Cli) -> AppConfig {
    if let Some(mode) = cli.mode {
        config.mode = mode;
    }
    let logger = Logger::new(config.mode);
    if let Some(output_mode) = &cli.output_mode {
        config.output_mode = output_mode.clone();
    } else if cli.display {
        config.output_mode = OutputMode::Display;
    } else if config.output_mode == OutputMode::Display {
        logger.warn(
            "Ignoring output_mode: Display from YAML; pass --display or --output-mode display to open the window",
        );
        config.output_mode = OutputMode::Headless;
    }
    if let Some(processing_type) = &cli.processing_type {
        config.algorithm_config.processing_type = processing_type.clone();
    }
    if let Some(camera_index) = cli.camera_index {
        config.camera_index = Some(camera_index);
    }
    if let Some(fps) = cli.fps {
        config.fps = fps;
    }
    if let Some(video_path) = &cli.video_path {
        config.video_path = Some(video_path.clone());
    }
    if let Some(resolution) = &cli.resolution {
        config.resolution = Some(*resolution);
    }
    if let Some(resolution) = &cli.display_resolution {
        config.display_resolution = Some(*resolution);
    }
    if let Some(resolution) = &cli.input_resolution {
        config.algorithm_config.input_resolution = Some(*resolution);
    }
    if let Some(value) = cli.fb_max {
        config.algorithm_config.fb_max = value;
    }
    if let Some(value) = cli.ransac_h {
        config.algorithm_config.ransac_h = value;
    }
    if let Some(value) = cli.tau_min {
        config.algorithm_config.tau_min = value;
    }
    if let Some(value) = cli.k_sigma {
        config.algorithm_config.k_sigma = value;
    }
    if let Some(value) = cli.tau_epi {
        config.algorithm_config.tau_epi = value;
    }
    if let Some(value) = cli.max_corners {
        config.algorithm_config.max_corners = value;
    }
    if let Some(value) = cli.quality_level {
        config.algorithm_config.quality_level = value;
    }
    if let Some(value) = cli.min_distance {
        config.algorithm_config.min_distance = value;
    }
    if let Some(value) = cli.min_inliers {
        config.algorithm_config.min_inliers = value;
    }
    if let Some(value) = cli.parallax_threshold {
        config.algorithm_config.parallax_threshold = value;
    }
    if let Some(value) = cli.flow_preset {
        config.algorithm_config.flow_preset = value;
    }
    if let Some(value) = cli.finest_scale {
        config.algorithm_config.finest_scale = value;
    }
    if let Some(value) = cli.morphology_open_size {
        config.algorithm_config.morphology_open_size = value;
    }
    if let Some(value) = cli.morphology_close_size {
        config.algorithm_config.morphology_close_size = value;
    }
    if let Some(value) = cli.min_component_area {
        config.algorithm_config.min_component_area = value;
    }
    if let Some(value) = cli.search_score_threshold {
        config.algorithm_config.search_score_threshold = value;
    }
    if let Some(value) = cli.tracking_score_threshold {
        config.algorithm_config.tracking_score_threshold = value;
    }
    if let Some(value) = cli.confirmation_window {
        config.algorithm_config.confirmation_window = value;
    }
    if let Some(value) = cli.confirmation_hits {
        config.algorithm_config.confirmation_hits = value;
    }
    if let Some(value) = cli.tracker_gate_scale {
        config.algorithm_config.tracker_gate_scale = value;
    }
    if let Some(value) = cli.max_coast_frames {
        config.algorithm_config.max_coast_frames = value;
    }
    if let Some(value) = cli.min_correlation {
        config.algorithm_config.min_correlation = value;
    }
    config
}

fn resolve_config(cli: &Cli) -> Result<AppConfig, String> {
    let fallback_logger = Logger::new(cli.mode.unwrap_or(AppConfig::default().mode));
    let config = apply_overrides(
        read_config(&cli.config_path).inspect_err(|error| fallback_logger.error(error))?,
        cli,
    );
    if let Err(error) = config.validate() {
        let error = format!("Invalid configuration: {error}");
        Logger::new(config.mode).error(&error);
        return Err(error);
    }
    Ok(config)
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let config = resolve_config(&cli)?;
    diff_motion_core::process_video_stream(config)
}

fn main() {
    if run().is_err() {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    const YAML_CONFIG: &str = r#"
algorithm_config:
  processing_type: motion
  input_resolution:
    width: 960
    height: 540
  tau_min: 1.25
output_mode: Headless
camera_index: 2
fps: 59.94
video_path: yaml.mp4
resolution:
  width: 1920
  height: 1080
display_resolution:
  width: 1280
  height: 720
"#;

    #[test]
    fn loads_default_config_file_when_no_path_is_provided() {
        let mut cli = Cli::try_parse_from(["diff-motion"]).unwrap();

        assert_eq!(cli.config_path, PathBuf::from("config/default.yaml"));

        cli.config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/default.yaml");

        let config = resolve_config(&cli).unwrap();

        assert_eq!(
            serde_yaml::to_value(config).unwrap(),
            serde_yaml::to_value(AppConfig::default()).unwrap()
        );
    }

    #[test]
    fn selected_partial_config_inherits_defaults_and_cli_overrides_it() {
        let path = std::env::temp_dir().join(format!(
            "diff-motion-custom-config-{}.yaml",
            std::process::id()
        ));
        fs::write(&path, "algorithm_config:\n  tau_min: 1.25\n").unwrap();
        let cli = Cli::try_parse_from([
            "diff-motion",
            "--config-path",
            path.to_str().unwrap(),
            "--tau-min",
            "1.75",
        ])
        .unwrap();

        let config = resolve_config(&cli).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(config.algorithm_config.processing_type, "motion");
        assert_eq!(config.algorithm_config.tau_min, 1.75);
        assert_eq!(config.camera_index, Some(0));
        assert_eq!(config.algorithm_config.max_corners, 400);
    }

    #[test]
    fn parses_complete_yaml_configuration() {
        let config = parse_config(YAML_CONFIG).unwrap();

        assert_eq!(config.algorithm_config.processing_type, "motion");
        assert_eq!(config.output_mode, OutputMode::Headless);
        assert_eq!(config.camera_index, Some(2));
        assert_eq!(config.fps, 59.94);
        assert_eq!(config.video_path.as_deref(), Some("yaml.mp4"));
        let resolution = config.resolution.unwrap();
        assert_eq!((resolution.width, resolution.height), (1920, 1080));
        let input_resolution = config.algorithm_config.input_resolution.unwrap();
        assert_eq!(
            (input_resolution.width, input_resolution.height),
            (960, 540)
        );
        assert_eq!(config.algorithm_config.tau_min, 1.25);
        let display_resolution = config.display_resolution.unwrap();
        assert_eq!(
            (display_resolution.width, display_resolution.height),
            (1280, 720)
        );
    }

    #[test]
    fn every_cli_value_overrides_the_shared_configuration() {
        let cli = Cli::try_parse_from([
            "diff-motion",
            "--processing-type",
            "motion",
            "--mode",
            "debug",
            "--output-mode",
            "display",
            "--camera-index",
            "4",
            "--fps",
            "60",
            "--video-path",
            "cli.mp4",
            "--resolution",
            "1280x720",
            "--display-resolution",
            "1024x576",
            "--input-resolution",
            "640x360",
            "--fb-max",
            "1.2",
            "--ransac-h",
            "2.0",
            "--tau-min",
            "0.9",
            "--k-sigma",
            "3.5",
            "--tau-epi",
            "1.1",
            "--max-corners",
            "500",
            "--quality-level",
            "0.02",
            "--min-distance",
            "6.0",
            "--min-inliers",
            "40",
            "--parallax-threshold",
            "0.2",
            "--flow-preset",
            "medium",
            "--finest-scale",
            "1",
            "--morphology-open-size",
            "3",
            "--morphology-close-size",
            "7",
            "--min-component-area",
            "80",
            "--search-score-threshold",
            "0.7",
            "--tracking-score-threshold",
            "0.4",
            "--confirmation-window",
            "5",
            "--confirmation-hits",
            "3",
            "--tracker-gate-scale",
            "1.5",
            "--max-coast-frames",
            "0",
            "--min-correlation",
            "0.3",
        ])
        .unwrap();
        let config = apply_overrides(AppConfig::default(), &cli);
        config.validate().unwrap();
        let value = serde_yaml::to_value(config).unwrap();

        let root_values = [
            ("mode", "debug"),
            ("output_mode", "Display"),
            ("camera_index", "4"),
            ("video_path", "cli.mp4"),
        ];
        for (key, expected) in root_values {
            assert_eq!(
                value[key],
                serde_yaml::from_str::<serde_yaml::Value>(expected).unwrap(),
                "wrong value for --{}",
                key.replace('_', "-")
            );
        }
        let algorithm = &value["algorithm_config"];
        assert_eq!(algorithm["processing_type"].as_str(), Some("motion"));
        assert_eq!(algorithm["flow_preset"].as_str(), Some("medium"));
        let algorithm_values = [
            ("fb_max", 1.2),
            ("ransac_h", 2.0),
            ("tau_min", 0.9),
            ("k_sigma", 3.5),
            ("tau_epi", 1.1),
            ("max_corners", 500.0),
            ("quality_level", 0.02),
            ("min_distance", 6.0),
            ("min_inliers", 40.0),
            ("parallax_threshold", 0.2),
            ("finest_scale", 1.0),
            ("morphology_open_size", 3.0),
            ("morphology_close_size", 7.0),
            ("min_component_area", 80.0),
            ("search_score_threshold", 0.7),
            ("tracking_score_threshold", 0.4),
            ("confirmation_window", 5.0),
            ("confirmation_hits", 3.0),
            ("tracker_gate_scale", 1.5),
            ("max_coast_frames", 0.0),
            ("min_correlation", 0.3),
        ];
        for (key, expected) in algorithm_values {
            let actual = algorithm[key].as_f64().unwrap();
            assert!(
                (actual - expected).abs() < 1e-6,
                "wrong value for --{}: expected {expected}, got {actual}",
                key.replace('_', "-")
            );
        }
        assert_eq!(value["resolution"]["width"].as_u64(), Some(1280));
        assert_eq!(value["resolution"]["height"].as_u64(), Some(720));
        assert_eq!(value["display_resolution"]["width"].as_u64(), Some(1024));
        assert_eq!(value["display_resolution"]["height"].as_u64(), Some(576));
        assert_eq!(value["fps"].as_f64(), Some(60.0));
        assert_eq!(algorithm["input_resolution"]["width"].as_u64(), Some(640));
        assert_eq!(algorithm["input_resolution"]["height"].as_u64(), Some(360));
    }

    #[test]
    fn absent_cli_values_preserve_yaml_configuration() {
        let cli = Cli::try_parse_from(["diff-motion"]).unwrap();
        let mut yaml_config = parse_config(YAML_CONFIG).unwrap();
        yaml_config.output_mode = OutputMode::JsonCamera;
        yaml_config.mode = Mode::Debug;

        let config = apply_overrides(yaml_config, &cli);

        assert_eq!(config.algorithm_config.processing_type, "motion");
        assert_eq!(config.output_mode, OutputMode::JsonCamera);
        assert_eq!(config.mode, Mode::Debug);
        assert_eq!(config.camera_index, Some(2));
        assert_eq!(config.fps, 59.94);
        assert_eq!(config.video_path.as_deref(), Some("yaml.mp4"));
        assert_eq!(config.algorithm_config.tau_min, 1.25);
        let resolution = config.resolution.unwrap();
        assert_eq!((resolution.width, resolution.height), (1920, 1080));
        let input_resolution = config.algorithm_config.input_resolution.unwrap();
        assert_eq!(
            (input_resolution.width, input_resolution.height),
            (960, 540)
        );
    }

    #[test]
    fn accepts_camera_index_boundaries_supported_by_opencv() {
        let maximum = i32::MAX.to_string();
        let cli = Cli::try_parse_from(["diff-motion", "--camera-index", "0"]).unwrap();
        let maximum_cli =
            Cli::try_parse_from(["diff-motion", "--camera-index", maximum.as_str()]).unwrap();

        assert_eq!(cli.camera_index, Some(0));
        assert_eq!(maximum_cli.camera_index, Some(i32::MAX as usize));
    }

    #[test]
    fn rejects_camera_index_outside_opencv_range() {
        let too_large = (i32::MAX as u64 + 1).to_string();

        let error =
            Cli::try_parse_from(["diff-motion", "--camera-index", too_large.as_str()]).unwrap_err();

        assert!(error.to_string().contains("OpenCV range"));
    }

    #[test]
    fn rejects_negative_camera_index() {
        assert!(Cli::try_parse_from(["diff-motion", "--camera-index=-1"]).is_err());
    }

    #[test]
    fn fps_inherits_yaml_and_cli_overrides_it() {
        let inherited = apply_overrides(
            parse_config("fps: 59.94\n").unwrap(),
            &Cli::try_parse_from(["diff-motion"]).unwrap(),
        );
        assert_eq!(inherited.fps, 59.94);

        let cli = Cli::try_parse_from(["diff-motion", "--fps", "60"]).unwrap();
        let overridden = apply_overrides(parse_config("fps: 29.97\n").unwrap(), &cli);
        assert_eq!(overridden.fps, 60.0);
    }

    #[test]
    fn rejects_non_positive_or_non_finite_fps() {
        for value in ["0", "-1", "NaN", "inf", "-inf"] {
            let argument = format!("--fps={value}");
            assert!(
                Cli::try_parse_from(["diff-motion", argument.as_str()]).is_err(),
                "accepted invalid FPS {value}"
            );
        }

        let cli = Cli::try_parse_from(["diff-motion", "--fps", "29.97"]).unwrap();
        assert_eq!(cli.fps, Some(29.97));
    }

    #[test]
    fn reports_malformed_config_file() {
        let path = std::env::temp_dir().join(format!(
            "diff-motion-malformed-config-{}.yaml",
            std::process::id()
        ));
        fs::write(&path, "output_mode: [").unwrap();
        let cli =
            Cli::try_parse_from(["diff-motion", "--config-path", path.to_str().unwrap()]).unwrap();

        let error = resolve_config(&cli).unwrap_err();
        fs::remove_file(path).unwrap();

        assert!(error.contains("Failed to parse config file"));
    }

    #[test]
    fn partial_yaml_inherits_defaults_and_unknown_fields_are_rejected() {
        let config = parse_config("algorithm_config:\n  tau_min: 1.5\n").unwrap();

        assert_eq!(config.algorithm_config.processing_type, "motion");
        assert_eq!(config.algorithm_config.tau_min, 1.5);
        assert_eq!(config.algorithm_config.max_corners, 400);
        assert_eq!(config.camera_index, Some(0));
        assert!(parse_config("algorithm_config:\n  unknown: 1\n").is_err());
        assert!(parse_config("unknown: 1\n").is_err());
    }

    #[test]
    fn reports_missing_config_file() {
        let path = std::env::temp_dir().join(format!(
            "diff-motion-missing-config-{}.yaml",
            std::process::id()
        ));
        let cli =
            Cli::try_parse_from(["diff-motion", "--config-path", path.to_str().unwrap()]).unwrap();

        let error = resolve_config(&cli).unwrap_err();

        assert!(error.contains("Failed to read config file"));
        assert!(error.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn accepts_config_path_and_legacy_aliases() {
        for option in ["--config-path", "--config", "-c"] {
            let cli = Cli::try_parse_from(["diff-motion", option, "config.yaml"]).unwrap();

            assert_eq!(cli.config_path, PathBuf::from("config.yaml"));
        }
    }

    #[test]
    fn config_help_shows_canonical_option_and_default_path() {
        let mut help = Vec::new();
        Cli::command().write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();

        assert!(help.contains("--config-path"));
        assert!(help.contains("[alias: --config]"));
        assert!(help.contains("[default: config/default.yaml]"));
        assert_eq!(help.matches("[default:").count(), 1);
        assert!(help.contains("default YAML: camera-selected mode"));
        assert!(help.contains("default YAML: 30"));
        assert!(help.contains("Ignored when a video file is supplied"));
        let resolution = AppConfig::default()
            .algorithm_config
            .resolved_input_resolution();
        assert!(help.contains(&format!(
            "default YAML: {}x{}",
            resolution.width, resolution.height
        )));
        assert!(help.contains("Omitted options use values from the selected YAML configuration"));
        assert!(help.contains("a display window opens only when --display"));
        assert!(help.contains("--mode"));
        assert!(help.contains("possible values: none, info, debug"));
        for option in [
            "--processing-type",
            "--display",
            "--output-mode",
            "--camera-index",
            "--fps",
            "--video-path",
            "--resolution",
            "--display-resolution",
            "--input-resolution",
            "--fb-max",
            "--ransac-h",
            "--tau-min",
            "--k-sigma",
            "--tau-epi",
            "--max-corners",
            "--quality-level",
            "--min-distance",
            "--min-inliers",
            "--parallax-threshold",
            "--flow-preset",
            "--finest-scale",
            "--morphology-open-size",
            "--morphology-close-size",
            "--min-component-area",
            "--search-score-threshold",
            "--tracking-score-threshold",
            "--confirmation-window",
            "--confirmation-hits",
            "--tracker-gate-scale",
            "--max-coast-frames",
            "--min-correlation",
        ] {
            assert!(help.contains(option), "help is missing {option}");
        }
    }

    #[test]
    fn accepts_only_literal_logging_modes() {
        for (value, expected) in [
            ("none", Mode::None),
            ("info", Mode::Info),
            ("debug", Mode::Debug),
        ] {
            let cli = Cli::try_parse_from(["diff-motion", "--mode", value]).unwrap();

            assert_eq!(cli.mode, Some(expected));
        }

        for value in ["INFO", "verbose", "off"] {
            assert!(Cli::try_parse_from(["diff-motion", "--mode", value]).is_err());
        }
    }

    #[test]
    fn logging_mode_inherits_yaml_and_cli_overrides_it() {
        let inherited = apply_overrides(
            parse_config("mode: debug\n").unwrap(),
            &Cli::try_parse_from(["diff-motion"]).unwrap(),
        );
        assert_eq!(inherited.mode, Mode::Debug);

        let cli = Cli::try_parse_from(["diff-motion", "--mode", "info"]).unwrap();
        let overridden = apply_overrides(parse_config("mode: none\n").unwrap(), &cli);
        assert_eq!(overridden.mode, Mode::Info);
    }

    #[test]
    fn parses_camera_and_algorithm_resolutions() {
        let cli = Cli::try_parse_from([
            "diff-motion",
            "--resolution",
            "1280x720",
            "--input-resolution",
            "640x360",
            "--display-resolution",
            "1024x576",
        ])
        .unwrap();

        let resolution = cli.resolution.unwrap();
        assert_eq!((resolution.width, resolution.height), (1280, 720));
        let input_resolution = cli.input_resolution.unwrap();
        assert_eq!(
            (input_resolution.width, input_resolution.height),
            (640, 360)
        );
        let display_resolution = cli.display_resolution.unwrap();
        assert_eq!(
            (display_resolution.width, display_resolution.height),
            (1024, 576)
        );
    }

    #[test]
    fn rejects_invalid_resolutions() {
        for value in [
            "640",
            "640X360",
            "640x360x1",
            "0x360",
            "640x0",
            "-1x360",
            "2147483648x360",
        ] {
            let argument = format!("--resolution={value}");
            let error = Cli::try_parse_from(["diff-motion", argument.as_str()]).unwrap_err();

            assert!(
                error.to_string().contains("resolution"),
                "unexpected error for {value}: {error}"
            );
        }
    }

    #[test]
    fn resolution_cli_values_override_yaml_independently() {
        let camera_cli = Cli::try_parse_from(["diff-motion", "--resolution", "1280x720"]).unwrap();
        let camera_config = apply_overrides(parse_config(YAML_CONFIG).unwrap(), &camera_cli);

        let camera_resolution = camera_config.resolution.unwrap();
        assert_eq!(
            (camera_resolution.width, camera_resolution.height),
            (1280, 720)
        );
        let input_resolution = camera_config.algorithm_config.input_resolution.unwrap();
        assert_eq!(
            (input_resolution.width, input_resolution.height),
            (960, 540)
        );

        let input_cli =
            Cli::try_parse_from(["diff-motion", "--input-resolution", "640x360"]).unwrap();
        let input_config = apply_overrides(parse_config(YAML_CONFIG).unwrap(), &input_cli);

        let camera_resolution = input_config.resolution.unwrap();
        assert_eq!(
            (camera_resolution.width, camera_resolution.height),
            (1920, 1080)
        );
        let input_resolution = input_config.algorithm_config.input_resolution.unwrap();
        assert_eq!(
            (input_resolution.width, input_resolution.height),
            (640, 360)
        );
    }

    #[test]
    fn display_alias_conflicts_with_output_mode() {
        let error = Cli::try_parse_from(["diff-motion", "--display", "--output-mode", "headless"])
            .unwrap_err();

        assert!(error.to_string().contains("cannot be used with"));
    }

    #[test]
    fn output_mode_can_disable_display_selected_by_yaml() {
        let cli = Cli::try_parse_from(["diff-motion", "--output-mode", "headless"]).unwrap();
        let config = apply_overrides(parse_config("output_mode: Display\n").unwrap(), &cli);

        assert_eq!(config.output_mode, OutputMode::Headless);
    }

    #[test]
    fn display_selected_by_yaml_requires_an_explicit_cli_flag() {
        let cli = Cli::try_parse_from(["diff-motion"]).unwrap();
        let config = apply_overrides(parse_config("output_mode: Display\n").unwrap(), &cli);

        assert_eq!(config.output_mode, OutputMode::Headless);
    }

    #[test]
    fn explicit_display_options_enable_the_window() {
        for args in [
            vec!["diff-motion", "--display"],
            vec!["diff-motion", "--output-mode", "display"],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            let config = apply_overrides(AppConfig::default(), &cli);

            assert_eq!(config.output_mode, OutputMode::Display);
        }
    }

    #[test]
    fn min_correlation_cli_value_reaches_algorithm_configuration() {
        let cli = Cli::try_parse_from(["diff-motion", "--min-correlation", "0.42"]).unwrap();
        let config = apply_overrides(AppConfig::default(), &cli);

        assert_eq!(config.algorithm_config.min_correlation, 0.42);
    }

    #[test]
    fn final_configuration_is_validated_after_cli_overrides() {
        let path = std::env::temp_dir().join(format!(
            "diff-motion-validation-config-{}.yaml",
            std::process::id()
        ));
        fs::write(&path, "algorithm_config:\n  processing_type: motion\n").unwrap();
        let cli = Cli::try_parse_from([
            "diff-motion",
            "--config-path",
            path.to_str().unwrap(),
            "--processing-type",
            "unsupported",
        ])
        .unwrap();

        let error = resolve_config(&cli).unwrap_err();
        fs::remove_file(path).unwrap();

        assert!(error.contains("Invalid configuration"));
        assert!(error.contains("Unsupported processing type"));
    }
}
