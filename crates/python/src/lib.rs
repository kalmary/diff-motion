#![allow(clippy::useless_conversion)]

use diff_motion_core::config::{AlgorithmConfig, AppConfig, OutputMode};
use diff_motion_core::process_video_stream_with_callback;
use diff_motion_core::types::ProcessingResult;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

fn result_to_python<'py>(
    py: Python<'py>,
    result: &ProcessingResult,
) -> PyResult<Bound<'py, PyDict>> {
    let mask = PyDict::new_bound(py);
    mask.set_item("data", PyBytes::new_bound(py, &result.mask.data))?;
    mask.set_item("width", result.mask.width)?;
    mask.set_item("height", result.mask.height)?;
    let objects = PyList::empty_bound(py);
    for object in &result.objects {
        let bbox = PyDict::new_bound(py);
        bbox.set_item("x", object.bbox.x)?;
        bbox.set_item("y", object.bbox.y)?;
        bbox.set_item("width", object.bbox.width)?;
        bbox.set_item("height", object.bbox.height)?;
        let item = PyDict::new_bound(py);
        item.set_item("class_id", object.class_id)?;
        item.set_item("class_name", &object.class_name)?;
        item.set_item("confidence", object.confidence)?;
        item.set_item("bbox", bbox)?;
        objects.append(item)?;
    }
    let output = PyDict::new_bound(py);
    output.set_item("mask", mask)?;
    output.set_item("objects", objects)?;
    Ok(output)
}

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
#[pyo3(signature = (headless=true, processing_type=None, camera_index=None, video_path=None, callback=None))]
fn run(
    py: Python<'_>,
    headless: bool,
    processing_type: Option<String>,
    camera_index: Option<usize>,
    video_path: Option<String>,
    callback: Option<Py<PyAny>>,
) -> PyResult<()> {
    if callback
        .as_ref()
        .is_some_and(|callback| !callback.bind(py).is_callable())
    {
        return Err(PyTypeError::new_err("callback must be callable"));
    }
    let config = build_config(headless, processing_type, camera_index, video_path);
    let mut callback_error = None;
    let processing = py.allow_threads(|| {
        process_video_stream_with_callback(config, |_, result| {
            if let Some(callback) = &callback {
                let received =
                    Python::with_gil(|py| callback.call1(py, (result_to_python(py, result)?,)));
                if let Err(error) = received {
                    let message = error.to_string();
                    callback_error = Some(error);
                    return Err(message);
                }
            }
            Ok(())
        })
    });
    if let Some(error) = callback_error {
        return Err(error);
    }
    processing.map_err(PyRuntimeError::new_err)
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
