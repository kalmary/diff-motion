use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_yaml::Value;

const DEFAULT_CONFIG: &str = include_str!("../../../config/default.yaml");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    pub algorithm_config: AlgorithmConfig,
    pub output_mode: OutputMode,
    pub mode: Mode,
    pub camera_index: Option<usize>,
    pub fps: f64,
    pub video_path: Option<String>,
    pub resolution: Option<Resolution>,
    pub display_resolution: Option<Resolution>,
}

impl AppConfig {
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        let mut config: Value = serde_yaml::from_str(DEFAULT_CONFIG)
            .map_err(|error| format!("Invalid embedded default configuration: {error}"))?;
        let overrides: Value = serde_yaml::from_str(yaml)
            .map_err(|error| format!("Invalid YAML configuration: {error}"))?;
        merge_yaml(&mut config, overrides);
        serde_yaml::from_value(config)
            .map_err(|error| format!("Invalid application configuration: {error}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        self.algorithm_config.validate()?;
        if !self.fps.is_finite() || self.fps <= 0.0 {
            return Err("fps must be finite and positive".to_string());
        }
        if let Some(index) = self.camera_index {
            i32::try_from(index)
                .map_err(|_| format!("Camera index {index} exceeds the OpenCV range"))?;
        }
        if let Some(resolution) = self.resolution {
            resolution.validate()?;
        }
        if let Some(resolution) = self.display_resolution {
            resolution.validate()?;
        }
        Ok(())
    }

    pub fn resolved_camera_index(&self) -> usize {
        self.camera_index.unwrap_or_else(|| {
            Self::default()
                .camera_index
                .expect("embedded default camera index must be set")
        })
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        serde_yaml::from_str(DEFAULT_CONFIG).expect("embedded default configuration must be valid")
    }
}

fn merge_yaml(base: &mut Value, overrides: Value) {
    match (base, overrides) {
        (Value::Mapping(base), Value::Mapping(overrides)) => {
            for (key, value) in overrides {
                if let Some(base_value) = base.get_mut(&key) {
                    merge_yaml(base_value, value);
                } else {
                    base.insert(key, value);
                }
            }
        }
        (base, value) => *base = value,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlgorithmConfig {
    pub processing_type: String,
    pub input_resolution: Option<Resolution>,
    pub fb_max: f32,
    pub ransac_h: f64,
    pub tau_min: f32,
    pub k_sigma: f32,
    pub tau_epi: f32,
    pub max_corners: i32,
    pub quality_level: f64,
    pub min_distance: f64,
    pub min_inliers: usize,
    pub parallax_threshold: f64,
    pub flow_preset: FlowPreset,
    pub finest_scale: i32,
    pub morphology_open_size: i32,
    pub morphology_close_size: i32,
    pub min_component_area: i32,
    pub search_score_threshold: f32,
    pub tracking_score_threshold: f32,
    pub confirmation_window: usize,
    pub confirmation_hits: usize,
    pub tracker_gate_scale: f32,
    /// Maximum template-propagated frames; zero disables mask propagation.
    pub max_coast_frames: u8,
    /// Appearance agreement required for motion observations and template coasting.
    pub min_correlation: f64,
}

impl AlgorithmConfig {
    pub fn resolved_input_resolution(&self) -> Resolution {
        self.input_resolution.unwrap_or_else(|| {
            Self::default()
                .input_resolution
                .expect("embedded default input resolution must be set")
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.processing_type.as_str(), "" | "motion") {
            return Err(format!(
                "Unsupported processing type '{}'; expected motion",
                self.processing_type
            ));
        }
        let input_resolution = self.resolved_input_resolution();
        input_resolution.validate_input()?;
        if !self.fb_max.is_finite() || self.fb_max <= 0.0 {
            return Err("fb_max must be finite and positive".to_string());
        }
        if !self.ransac_h.is_finite() || self.ransac_h <= 0.0 {
            return Err("ransac_h must be finite and positive".to_string());
        }
        if !self.tau_min.is_finite() || self.tau_min <= 0.0 {
            return Err("tau_min must be finite and positive".to_string());
        }
        if !self.k_sigma.is_finite() || self.k_sigma < 0.0 {
            return Err("k_sigma must be finite and non-negative".to_string());
        }
        if !self.tau_epi.is_finite() || self.tau_epi <= 0.0 {
            return Err("tau_epi must be finite and positive".to_string());
        }
        if self.max_corners < 8 {
            return Err("max_corners must be at least 8".to_string());
        }
        if !self.quality_level.is_finite() || self.quality_level <= 0.0 || self.quality_level > 1.0
        {
            return Err("quality_level must be finite and in (0, 1]".to_string());
        }
        if !self.min_distance.is_finite() || self.min_distance < 0.0 {
            return Err("min_distance must be finite and non-negative".to_string());
        }
        if self.min_inliers < 8 || self.min_inliers > self.max_corners as usize {
            return Err("min_inliers must be between 8 and max_corners".to_string());
        }
        if !self.parallax_threshold.is_finite() || !(0.0..=1.0).contains(&self.parallax_threshold) {
            return Err("parallax_threshold must be finite and in [0, 1]".to_string());
        }
        let max_dimension = f64::from(input_resolution.width.max(input_resolution.height));
        let min_dimension = f64::from(input_resolution.width.min(input_resolution.height));
        let max_finest_scale = ((max_dimension / 32.0).log2() + 0.5) as i32;
        let max_finest_scale = max_finest_scale.min((min_dimension / 8.0).log2() as i32);
        if self.finest_scale < 0 || self.finest_scale > max_finest_scale {
            return Err(format!(
                "finest_scale must be between 0 and {max_finest_scale} for input resolution {}x{}",
                input_resolution.width, input_resolution.height
            ));
        }
        for (name, size) in [
            ("morphology_open_size", self.morphology_open_size),
            ("morphology_close_size", self.morphology_close_size),
        ] {
            if size <= 0 || size % 2 == 0 {
                return Err(format!("{name} must be positive and odd"));
            }
        }
        if self.min_component_area <= 0 {
            return Err("min_component_area must be positive".to_string());
        }
        if !self.search_score_threshold.is_finite() || self.search_score_threshold <= 0.0 {
            return Err("search_score_threshold must be finite and positive".to_string());
        }
        if !self.tracking_score_threshold.is_finite()
            || self.tracking_score_threshold <= 0.0
            || self.tracking_score_threshold > self.search_score_threshold
        {
            return Err(
                "tracking_score_threshold must be finite, positive, and no greater than search_score_threshold"
                    .to_string(),
            );
        }
        if self.confirmation_window == 0
            || self.confirmation_hits == 0
            || self.confirmation_hits > self.confirmation_window
        {
            return Err(
                "confirmation_hits and confirmation_window must be positive, with hits no greater than the window"
                    .to_string(),
            );
        }
        if !self.tracker_gate_scale.is_finite() || self.tracker_gate_scale < 1.0 {
            return Err("tracker_gate_scale must be finite and at least 1".to_string());
        }
        if !self.min_correlation.is_finite() || !(-1.0..=1.0).contains(&self.min_correlation) {
            return Err("min_correlation must be finite and in [-1, 1]".to_string());
        }
        Ok(())
    }
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        AppConfig::default().algorithm_config
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FlowPreset {
    Ultrafast,
    Fast,
    Medium,
}

impl FromStr for FlowPreset {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "ultrafast" => Ok(Self::Ultrafast),
            "fast" => Ok(Self::Fast),
            "medium" => Ok(Self::Medium),
            _ => Err(format!(
                "Unsupported flow preset '{value}'; expected ultrafast, fast, or medium"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    pub fn validate_input(&self) -> Result<(), String> {
        self.validate()?;
        if self.width.min(self.height) < 8 || self.width.max(self.height) < 12 {
            return Err("Algorithm input needs both dimensions >= 8 and at least one >= 12 for optical flow".to_string());
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0
            || self.height == 0
            || self.width > i32::MAX as u32
            || self.height > i32::MAX as u32
        {
            return Err(
                "Resolution dimensions must be positive and within the OpenCV range".to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OutputMode {
    Headless,
    Display,
    JsonCamera,
}

impl FromStr for OutputMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "headless" => Ok(Self::Headless),
            "display" => Ok(Self::Display),
            "jsoncamera" | "json-camera" | "json_camera" => Ok(Self::JsonCamera),
            _ => Err(format!(
                "Unsupported output mode '{value}'; expected headless, display, or json-camera"
            )),
        }
    }
}

impl Default for OutputMode {
    fn default() -> Self {
        AppConfig::default().output_mode
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    None,
    Info,
    Debug,
}

impl FromStr for Mode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            _ => Err(format!(
                "Unsupported logging mode '{value}'; expected none, info, or debug"
            )),
        }
    }
}

impl Default for Mode {
    fn default() -> Self {
        AppConfig::default().mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(
            serde_yaml::to_value(&config).unwrap(),
            serde_yaml::to_value(AppConfig::from_yaml("{}").unwrap()).unwrap()
        );
        config.validate().unwrap();
    }

    #[test]
    fn logging_mode_uses_the_same_literals_in_yaml_and_arguments() {
        for (literal, mode) in [
            ("none", Mode::None),
            ("info", Mode::Info),
            ("debug", Mode::Debug),
        ] {
            assert_eq!(literal.parse::<Mode>().unwrap(), mode);
            assert_eq!(
                AppConfig::from_yaml(&format!("mode: {literal}"))
                    .unwrap()
                    .mode,
                mode
            );
        }
        for literal in ["DEBUG", "quiet", "null", ""] {
            assert!(literal.parse::<Mode>().is_err());
            assert!(AppConfig::from_yaml(&format!("mode: {literal}")).is_err());
        }
    }

    #[test]
    fn fps_defaults_merges_and_validation_are_shared() {
        assert_eq!(AppConfig::from_yaml("{}").unwrap().fps, 30.0);
        let config = AppConfig::from_yaml("fps: 59.94").unwrap();
        assert_eq!(config.fps, 59.94);
        config.validate().unwrap();
        for fps in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let config = AppConfig {
                fps,
                ..AppConfig::default()
            };
            assert!(config.validate().unwrap_err().contains("fps"));
        }
    }

    #[test]
    fn partial_yaml_inherits_canonical_defaults_and_null_replaces_values() {
        let config = AppConfig::from_yaml(
            r#"
camera_index: null
algorithm_config:
  tau_min: 1.25
"#,
        )
        .unwrap();

        assert_eq!(config.camera_index, None);
        assert_eq!(config.algorithm_config.tau_min, 1.25);
        assert_eq!(config.algorithm_config.max_corners, 400);
        assert_eq!(config.algorithm_config.flow_preset, FlowPreset::Fast);
    }

    #[test]
    fn yaml_parsing_defers_range_validation_until_after_overrides() {
        let config = AppConfig::from_yaml("algorithm_config:\n  tau_min: -1").unwrap();

        assert!(config.validate().is_err());
    }

    #[test]
    fn null_input_resolution_uses_the_canonical_processing_size() {
        let config = AppConfig::from_yaml("algorithm_config:\n  input_resolution: null").unwrap();

        assert_eq!(
            config.algorithm_config.resolved_input_resolution(),
            AlgorithmConfig::default()
                .input_resolution
                .expect("canonical input resolution must be set")
        );
    }

    #[test]
    fn null_camera_index_uses_the_canonical_camera() {
        let config = AppConfig::from_yaml("camera_index: null").unwrap();

        assert_eq!(config.resolved_camera_index(), 0);
    }

    #[test]
    fn yaml_rejects_unknown_keys_at_every_level() {
        assert!(AppConfig::from_yaml("unknown: 1").is_err());
        assert!(AppConfig::from_yaml("algorithm_config:\n  unknown: 1").is_err());
    }

    #[test]
    fn zero_coast_frames_disables_mask_propagation() {
        let config = AppConfig::from_yaml("algorithm_config:\n  max_coast_frames: 0").unwrap();
        assert_eq!(config.algorithm_config.max_coast_frames, 0);
        config.validate().unwrap();
    }

    #[test]
    fn rejects_invalid_algorithm_ranges() {
        let mut config = AppConfig::default();
        config.algorithm_config.min_inliers = 7;
        assert!(config.validate().is_err());

        let mut config = AppConfig::default();
        config.algorithm_config.confirmation_hits = 6;
        assert!(config.validate().is_err());

        let mut config = AppConfig::default();
        config.algorithm_config.morphology_open_size = 4;
        assert!(config.validate().is_err());

        let mut config = AppConfig::default();
        config.algorithm_config.tracking_score_threshold = 1.1;
        assert!(config.validate().is_err());

        let mut config = AppConfig::default();
        config.algorithm_config.finest_scale = 5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn parses_flow_presets_case_insensitively_for_cli_use() {
        assert_eq!("ultrafast".parse(), Ok(FlowPreset::Ultrafast));
        assert_eq!("FAST".parse(), Ok(FlowPreset::Fast));
        assert_eq!("Medium".parse(), Ok(FlowPreset::Medium));
        assert!("slow".parse::<FlowPreset>().is_err());
    }

    #[test]
    fn parses_output_modes_case_insensitively_for_cli_use() {
        assert_eq!("HEADLESS".parse(), Ok(OutputMode::Headless));
        assert_eq!("display".parse(), Ok(OutputMode::Display));
        assert_eq!("json-camera".parse(), Ok(OutputMode::JsonCamera));
        assert!("window".parse::<OutputMode>().is_err());
    }

    #[test]
    fn rejects_input_sizes_too_small_for_optical_flow() {
        for (width, height) in [(12, 8), (8, 12), (320, 180)] {
            assert!(Resolution { width, height }.validate_input().is_ok());
        }
        for (width, height) in [(11, 11), (12, 7), (7, 12)] {
            assert!(Resolution { width, height }.validate_input().is_err());
        }
    }

    #[test]
    fn validates_resolution_dimensions() {
        assert!(Resolution {
            width: 1280,
            height: 720
        }
        .validate()
        .is_ok());
        for (width, height) in [(0, 720), (1280, 0), (u32::MAX, 720)] {
            assert!(Resolution { width, height }.validate().is_err());
        }
    }
}
