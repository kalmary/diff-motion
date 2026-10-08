use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    pub algorithm_config: AlgorithmConfig,
    pub output_mode: OutputMode,
    pub camera_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlgorithmConfig {
    pub processing_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum OutputMode {
    #[default]
    Headless,
    Display,
    JsonCamera,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.output_mode, OutputMode::Headless);
        assert_eq!(config.camera_index, None);
        assert_eq!(config.algorithm_config.yolo_model_size, "");
    }
}
