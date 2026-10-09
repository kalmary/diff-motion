#![allow(clippy::useless_conversion)]

use diff_motion_core::config::{AlgorithmConfig, AppConfig, OutputMode};
use diff_motion_core::process_video_stream;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn build_config(
    headless: bool,
    processing_type: Option<String>,
    camera_index: Option<usize>,
    video_path: Option<String>,
) -> AppConfig {
    AppConfig {
        algorithm_config: AlgorithmConfig {
            processing_type: processing_type.unwrap_or_default(),
        },
        output_mode: if headless {
            OutputMode::Headless
        } else {
            OutputMode::Display
        },
        camera_index,
        video_path,
    }
}

#[pyfunction]
#[pyo3(signature = (headless=true, processing_type=None, camera_index=None, video_path=None))]
fn run(
    py: Python<'_>,
    headless: bool,
    processing_type: Option<String>,
    camera_index: Option<usize>,
    video_path: Option<String>,
) -> PyResult<()> {
    let config = build_config(headless, processing_type, camera_index, video_path);
    py.allow_threads(move || process_video_stream(config))
        .map_err(PyRuntimeError::new_err)
}

#[pymodule]
fn diff_motion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_headless_config() {
        let config = build_config(true, Some("motion".to_string()), Some(2), None);

        assert_eq!(config.output_mode, OutputMode::Headless);
        assert_eq!(config.algorithm_config.processing_type, "motion");
        assert_eq!(config.camera_index, Some(2));
        assert_eq!(config.video_path, None);
    }

    #[test]
    fn builds_display_config_with_video() {
        let config = build_config(false, None, None, Some("sample.mp4".to_string()));

        assert_eq!(config.output_mode, OutputMode::Display);
        assert_eq!(config.video_path.as_deref(), Some("sample.mp4"));
    }
}
