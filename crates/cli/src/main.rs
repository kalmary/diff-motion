use clap::Parser;
use diff_motion_core::config::{AppConfig, AlgorithmConfig, OutputMode};
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to config.yaml file
    #[arg(short, long)]
    pub config: Option<String>,

    /// Override processing type
    #[arg(long)]
    pub processing_type: Option<String>,

    /// Display processed video window
    #[arg(long)]
    pub display: bool,

    /// Camera index
    #[arg(long)]
    pub camera_index: Option<usize>,
}

fn main() {
    let cli = Cli::parse();

    let mut app_config = if let Some(path) = &cli.config {
        let content = fs::read_to_string(path).expect("Failed to read config file");
        serde_yaml::from_str(&content).expect("Failed to parse config file")
    } else {
        AppConfig::default()
    };

    if cli.display {
        app_config.output_mode = OutputMode::Display;
    }
    
    if let Some(pt) = cli.processing_type {
        app_config.algorithm_config.processing_type = pt;
    }

    if let Some(idx) = cli.camera_index {
        app_config.camera_index = Some(idx);
    }

    println!("Starting diff-motion with config: {:?}", app_config);
    
    diff_motion_core::process_video_stream(app_config);
}
