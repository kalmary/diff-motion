use pyo3::prelude::*;
use pyo3::types::PyDict;
use diff_motion_core::process_video_stream;
use diff_motion_core::config::{AppConfig, AlgorithmConfig, OutputMode};

#[pyfunction]
#[pyo3(signature = (headless=true, processing_type=None, camera_index=None, video_path=None))]
fn run(headless: bool, processing_type: Option<String>, camera_index: Option<usize>, video_path: Option<String>) {
    let mut config = AppConfig::default();
    
    if headless {
        config.output_mode = OutputMode::Headless;
    } else {
        config.output_mode = OutputMode::Display;
    }

    if let Some(pt) = processing_type {
        config.algorithm_config.processing_type = pt;
    }

    config.camera_index = camera_index;
    config.video_path = video_path;

    process_video_stream(config);
}

#[pymodule]
fn diff_motion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
