use clap::Parser;
use diff_motion_core::config::{AppConfig, OutputMode};
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to config.yaml file
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Override processing type
    #[arg(long)]
    pub processing_type: Option<String>,

    /// Display processed video window
    #[arg(long)]
    pub display: bool,

    /// Camera index
    #[arg(long, value_parser = parse_camera_index)]
    pub camera_index: Option<usize>,

    /// Path to a video file for testing
    #[arg(long)]
    pub video_path: Option<String>,
}

fn parse_config(content: &str) -> Result<AppConfig, String> {
    serde_yaml::from_str(content).map_err(|error| format!("Failed to parse config file: {error}"))
}

fn read_config(path: &Path) -> Result<AppConfig, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read config file {}: {error}", path.display()))?;
    parse_config(&content)
}

fn apply_overrides(mut config: AppConfig, cli: &Cli) -> AppConfig {
    if cli.display {
        config.output_mode = OutputMode::Display;
    }
    if let Some(processing_type) = &cli.processing_type {
        config.algorithm_config.processing_type = processing_type.clone();
    }
    if let Some(camera_index) = cli.camera_index {
        config.camera_index = Some(camera_index);
    }
    if let Some(video_path) = &cli.video_path {
        config.video_path = Some(video_path.clone());
    }
    config
}

fn resolve_config(cli: &Cli) -> Result<AppConfig, String> {
    let config = if let Some(path) = &cli.config {
        read_config(path)?
    } else {
        AppConfig::default()
    };
    Ok(apply_overrides(config, cli))
}

fn run() -> Result<(), String> {
    let config = resolve_config(&Cli::parse())?;
    println!("Starting diff-motion with config: {config:?}");
    diff_motion_core::process_video_stream(config)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML_CONFIG: &str = r#"
algorithm_config:
  processing_type: yaml-motion
output_mode: Headless
camera_index: 2
video_path: yaml.mp4
"#;

    #[test]
    fn uses_defaults_when_no_configuration_is_provided() {
        let cli = Cli::try_parse_from(["diff-motion"]).unwrap();

        let config = resolve_config(&cli).unwrap();

        assert_eq!(config.algorithm_config.processing_type, "");
        assert_eq!(config.output_mode, OutputMode::Headless);
        assert_eq!(config.camera_index, None);
        assert_eq!(config.video_path, None);
    }

    #[test]
    fn parses_complete_yaml_configuration() {
        let config = parse_config(YAML_CONFIG).unwrap();

        assert_eq!(config.algorithm_config.processing_type, "yaml-motion");
        assert_eq!(config.output_mode, OutputMode::Headless);
        assert_eq!(config.camera_index, Some(2));
        assert_eq!(config.video_path.as_deref(), Some("yaml.mp4"));
    }

    #[test]
    fn cli_values_override_yaml_configuration() {
        let cli = Cli::try_parse_from([
            "diff-motion",
            "--processing-type",
            "cli-motion",
            "--display",
            "--camera-index",
            "4",
            "--video-path",
            "cli.mp4",
        ])
        .unwrap();
        let yaml_config = parse_config(YAML_CONFIG).unwrap();

        let config = apply_overrides(yaml_config, &cli);

        assert_eq!(config.algorithm_config.processing_type, "cli-motion");
        assert_eq!(config.output_mode, OutputMode::Display);
        assert_eq!(config.camera_index, Some(4));
        assert_eq!(config.video_path.as_deref(), Some("cli.mp4"));
    }

    #[test]
    fn absent_cli_values_preserve_yaml_configuration() {
        let cli = Cli::try_parse_from(["diff-motion"]).unwrap();
        let mut yaml_config = parse_config(YAML_CONFIG).unwrap();
        yaml_config.output_mode = OutputMode::JsonCamera;

        let config = apply_overrides(yaml_config, &cli);

        assert_eq!(config.algorithm_config.processing_type, "yaml-motion");
        assert_eq!(config.output_mode, OutputMode::JsonCamera);
        assert_eq!(config.camera_index, Some(2));
        assert_eq!(config.video_path.as_deref(), Some("yaml.mp4"));
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
    fn reports_malformed_yaml() {
        let error = parse_config("output_mode: [").unwrap_err();

        assert!(error.contains("Failed to parse config file"));
    }

    #[test]
    fn reports_missing_config_file() {
        let path = std::env::temp_dir().join(format!(
            "diff-motion-missing-config-{}.yaml",
            std::process::id()
        ));

        let error = read_config(&path).unwrap_err();

        assert!(error.contains("Failed to read config file"));
        assert!(error.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn accepts_platform_native_config_paths() {
        let cli = Cli::try_parse_from(["diff-motion", "--config", "config.yaml"]).unwrap();

        assert_eq!(cli.config.as_deref(), Some(Path::new("config.yaml")));
    }
}
