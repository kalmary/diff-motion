#![allow(clippy::useless_conversion)]

use std::fs;

use diff_motion_core::config::{
    AlgorithmConfig, AppConfig, FlowPreset, Mode, OutputMode, Resolution,
};
use diff_motion_core::types::ProcessingResult;
use diff_motion_core::{logging, process_video_stream_with_callback};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

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

fn parse_resolution(value: Option<(i64, i64)>, name: &str) -> Result<Option<Resolution>, String> {
    let Some((width, height)) = value else {
        return Ok(None);
    };
    let invalid = || format!("{name} dimensions are invalid");
    let resolution = Resolution {
        width: u32::try_from(width).map_err(|_| invalid())?,
        height: u32::try_from(height).map_err(|_| invalid())?,
    };
    resolution
        .validate()
        .map_err(|error| format!("{name}: {error}"))?;
    Ok(Some(resolution))
}

fn parse_python_resolution(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<(i64, i64)>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let invalid = || PyValueError::new_err(format!("{name} must be a valid (width, height) pair"));
    let value = value.downcast::<PyTuple>().map_err(|_| invalid())?;
    if value.len() != 2 {
        return Err(invalid());
    }
    let width = value.get_item(0)?.extract().map_err(|_| invalid())?;
    let height = value.get_item(1)?.extract().map_err(|_| invalid())?;
    Ok(Some((width, height)))
}

#[derive(Default)]
struct AlgorithmOverrides {
    fb_max: Option<f32>,
    ransac_h: Option<f64>,
    tau_min: Option<f32>,
    k_sigma: Option<f32>,
    tau_epi: Option<f32>,
    max_corners: Option<i32>,
    quality_level: Option<f64>,
    min_distance: Option<f64>,
    min_inliers: Option<usize>,
    parallax_threshold: Option<f64>,
    flow_preset: Option<String>,
    finest_scale: Option<i32>,
    morphology_open_size: Option<i32>,
    morphology_close_size: Option<i32>,
    min_component_area: Option<i32>,
    search_score_threshold: Option<f32>,
    tracking_score_threshold: Option<f32>,
    confirmation_window: Option<usize>,
    confirmation_hits: Option<usize>,
    tracker_gate_scale: Option<f32>,
    max_coast_frames: Option<u8>,
    min_correlation: Option<f64>,
}

impl AlgorithmOverrides {
    fn apply(self, config: &mut AlgorithmConfig) -> Result<(), String> {
        if let Some(value) = self.fb_max {
            config.fb_max = value;
        }
        if let Some(value) = self.ransac_h {
            config.ransac_h = value;
        }
        if let Some(value) = self.tau_min {
            config.tau_min = value;
        }
        if let Some(value) = self.k_sigma {
            config.k_sigma = value;
        }
        if let Some(value) = self.tau_epi {
            config.tau_epi = value;
        }
        if let Some(value) = self.max_corners {
            config.max_corners = value;
        }
        if let Some(value) = self.quality_level {
            config.quality_level = value;
        }
        if let Some(value) = self.min_distance {
            config.min_distance = value;
        }
        if let Some(value) = self.min_inliers {
            config.min_inliers = value;
        }
        if let Some(value) = self.parallax_threshold {
            config.parallax_threshold = value;
        }
        if let Some(value) = self.flow_preset {
            config.flow_preset = value.parse::<FlowPreset>()?;
        }
        if let Some(value) = self.finest_scale {
            config.finest_scale = value;
        }
        if let Some(value) = self.morphology_open_size {
            config.morphology_open_size = value;
        }
        if let Some(value) = self.morphology_close_size {
            config.morphology_close_size = value;
        }
        if let Some(value) = self.min_component_area {
            config.min_component_area = value;
        }
        if let Some(value) = self.search_score_threshold {
            config.search_score_threshold = value;
        }
        if let Some(value) = self.tracking_score_threshold {
            config.tracking_score_threshold = value;
        }
        if let Some(value) = self.confirmation_window {
            config.confirmation_window = value;
        }
        if let Some(value) = self.confirmation_hits {
            config.confirmation_hits = value;
        }
        if let Some(value) = self.tracker_gate_scale {
            config.tracker_gate_scale = value;
        }
        if let Some(value) = self.max_coast_frames {
            config.max_coast_frames = value;
        }
        if let Some(value) = self.min_correlation {
            config.min_correlation = value;
        }
        Ok(())
    }
}

fn read_config(path: Option<&str>) -> PyResult<AppConfig> {
    let Some(path) = path else {
        return Ok(AppConfig::default());
    };
    let content = fs::read_to_string(path).map_err(|error| {
        PyRuntimeError::new_err(format!("Failed to read config file {path}: {error}"))
    })?;
    AppConfig::from_yaml(&content).map_err(|error| {
        PyValueError::new_err(format!("Invalid configuration file {path}: {error}"))
    })
}

#[allow(clippy::too_many_arguments)]
fn build_config(
    mut config: AppConfig,
    headless: Option<bool>,
    processing_type: Option<String>,
    camera_index: Option<usize>,
    video_path: Option<String>,
    output_mode: Option<String>,
    mode: Option<Mode>,
    resolution: Option<(i64, i64)>,
    input_resolution: Option<(i64, i64)>,
    display_resolution: Option<(i64, i64)>,
    fps: Option<f64>,
    algorithm_overrides: AlgorithmOverrides,
) -> Result<AppConfig, String> {
    if headless.is_some() && output_mode.is_some() {
        return Err("headless and output_mode cannot be used together".to_string());
    }
    if let Some(headless) = headless {
        config.output_mode = if headless {
            OutputMode::Headless
        } else {
            OutputMode::Display
        };
    }
    if let Some(output_mode) = output_mode {
        config.output_mode = output_mode.parse::<OutputMode>()?;
    }
    if let Some(mode) = mode {
        config.mode = mode;
    }
    if let Some(processing_type) = processing_type {
        config.algorithm_config.processing_type = processing_type;
    }
    if let Some(camera_index) = camera_index {
        config.camera_index = Some(camera_index);
    }
    if let Some(video_path) = video_path {
        config.video_path = Some(video_path);
    }
    if resolution.is_some() {
        config.resolution = parse_resolution(resolution, "resolution")?;
    }
    if input_resolution.is_some() {
        let input_resolution = parse_resolution(input_resolution, "input_resolution")?;
        if let Some(input_resolution) = input_resolution {
            input_resolution
                .validate_input()
                .map_err(|error| format!("input_resolution: {error}"))?;
        }
        config.algorithm_config.input_resolution = input_resolution;
    }
    if display_resolution.is_some() {
        config.display_resolution = parse_resolution(display_resolution, "display_resolution")?;
    }
    if let Some(fps) = fps {
        config.fps = fps;
    }
    algorithm_overrides.apply(&mut config.algorithm_config)?;
    config.validate()?;
    Ok(config)
}

/// Process a video stream. Set max_coast_frames=0 to disable propagated masks.
/// min_correlation checks appearance before accepting observations or coasting.
/// Logging info includes reliability and errors; debug also includes warnings.
#[pyfunction]
#[pyo3(signature = (headless=None, processing_type=None, camera_index=None, video_path=None, callback=None, *, config_path=None, mode=None, output_mode=None, resolution=None, input_resolution=None, display_resolution=None, fps=None, fb_max=None, ransac_h=None, tau_min=None, k_sigma=None, tau_epi=None, max_corners=None, quality_level=None, min_distance=None, min_inliers=None, parallax_threshold=None, flow_preset=None, finest_scale=None, morphology_open_size=None, morphology_close_size=None, min_component_area=None, search_score_threshold=None, tracking_score_threshold=None, confirmation_window=None, confirmation_hits=None, tracker_gate_scale=None, max_coast_frames=None, min_correlation=None))]
#[allow(clippy::too_many_arguments)]
fn run(
    py: Python<'_>,
    headless: Option<bool>,
    processing_type: Option<String>,
    camera_index: Option<usize>,
    video_path: Option<String>,
    callback: Option<Py<PyAny>>,
    config_path: Option<String>,
    mode: Option<String>,
    output_mode: Option<String>,
    resolution: Option<&Bound<'_, PyAny>>,
    input_resolution: Option<&Bound<'_, PyAny>>,
    display_resolution: Option<&Bound<'_, PyAny>>,
    fps: Option<f64>,
    fb_max: Option<f32>,
    ransac_h: Option<f64>,
    tau_min: Option<f32>,
    k_sigma: Option<f32>,
    tau_epi: Option<f32>,
    max_corners: Option<i32>,
    quality_level: Option<f64>,
    min_distance: Option<f64>,
    min_inliers: Option<usize>,
    parallax_threshold: Option<f64>,
    flow_preset: Option<String>,
    finest_scale: Option<i32>,
    morphology_open_size: Option<i32>,
    morphology_close_size: Option<i32>,
    min_component_area: Option<i32>,
    search_score_threshold: Option<f32>,
    tracking_score_threshold: Option<f32>,
    confirmation_window: Option<usize>,
    confirmation_hits: Option<usize>,
    tracker_gate_scale: Option<f32>,
    max_coast_frames: Option<u8>,
    min_correlation: Option<f64>,
) -> PyResult<()> {
    let mode = match mode.map(|value| value.parse::<Mode>()).transpose() {
        Ok(mode) => mode,
        Err(error) => {
            logging::Logger::new(Mode::default()).error(&error);
            return Err(PyValueError::new_err(error));
        }
    };
    let config = match read_config(config_path.as_deref()) {
        Ok(config) => config,
        Err(error) => {
            logging::Logger::new(mode.unwrap_or_default()).error(&error);
            return Err(error);
        }
    };
    let logger = logging::Logger::new(mode.unwrap_or(config.mode));
    let config = (|| -> PyResult<AppConfig> {
        if callback
            .as_ref()
            .is_some_and(|callback| !callback.bind(py).is_callable())
        {
            return Err(PyTypeError::new_err("callback must be callable"));
        }
        build_config(
            config,
            headless,
            processing_type,
            camera_index,
            video_path,
            output_mode,
            mode,
            parse_python_resolution(resolution, "resolution")?,
            parse_python_resolution(input_resolution, "input_resolution")?,
            parse_python_resolution(display_resolution, "display_resolution")?,
            fps,
            AlgorithmOverrides {
                fb_max,
                ransac_h,
                tau_min,
                k_sigma,
                tau_epi,
                max_corners,
                quality_level,
                min_distance,
                min_inliers,
                parallax_threshold,
                flow_preset,
                finest_scale,
                morphology_open_size,
                morphology_close_size,
                min_component_area,
                search_score_threshold,
                tracking_score_threshold,
                confirmation_window,
                confirmation_hits,
                tracker_gate_scale,
                max_coast_frames,
                min_correlation,
            },
        )
        .map_err(PyValueError::new_err)
    })()
    .inspect_err(|error| logger.error(error))?;
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
    fn builds_shared_default_config() {
        let config = build_config(
            AppConfig::default(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            AlgorithmOverrides::default(),
        )
        .unwrap();

        let expected = AppConfig::default();
        assert_eq!(config.output_mode, expected.output_mode);
        assert_eq!(config.mode, expected.mode);
        assert_eq!(config.fps, expected.fps);
        assert_eq!(
            config.algorithm_config.processing_type,
            expected.algorithm_config.processing_type
        );
        assert_eq!(
            config.algorithm_config.input_resolution,
            expected.algorithm_config.input_resolution
        );
        assert_eq!(config.camera_index, expected.camera_index);
        assert_eq!(config.video_path, expected.video_path);
        assert_eq!(config.resolution, expected.resolution);
        assert_eq!(
            config.algorithm_config.fb_max,
            expected.algorithm_config.fb_max
        );
        assert_eq!(
            config.algorithm_config.flow_preset,
            expected.algorithm_config.flow_preset
        );
    }

    #[test]
    fn positional_arguments_and_keywords_override_config() {
        let config = build_config(
            AppConfig::default(),
            Some(false),
            Some("motion".to_string()),
            Some(2),
            Some("sample.mp4".to_string()),
            None,
            Some(Mode::Debug),
            Some((1280, 720)),
            Some((320, 180)),
            None,
            Some(60.0),
            AlgorithmOverrides {
                tau_min: Some(1.25),
                flow_preset: Some("medium".to_string()),
                ..AlgorithmOverrides::default()
            },
        )
        .unwrap();

        assert_eq!(config.output_mode, OutputMode::Display);
        assert_eq!(config.mode, Mode::Debug);
        assert_eq!(config.fps, 60.0);
        assert_eq!(config.camera_index, Some(2));
        assert_eq!(config.video_path.as_deref(), Some("sample.mp4"));
        assert_eq!(config.resolution.unwrap().width, 1280);
        assert_eq!(config.algorithm_config.input_resolution.unwrap().width, 320);
        assert_eq!(config.algorithm_config.tau_min, 1.25);
        assert_eq!(config.algorithm_config.flow_preset, FlowPreset::Medium);
    }

    #[test]
    fn omitted_arguments_preserve_loaded_config() {
        let loaded = AppConfig {
            output_mode: OutputMode::Display,
            mode: Mode::Debug,
            fps: 29.97,
            camera_index: Some(3),
            algorithm_config: AlgorithmConfig {
                input_resolution: Some(Resolution {
                    width: 320,
                    height: 180,
                }),
                ..AlgorithmConfig::default()
            },
            ..AppConfig::default()
        };

        let config = build_config(
            loaded,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            AlgorithmOverrides::default(),
        )
        .unwrap();

        assert_eq!(config.output_mode, OutputMode::Display);
        assert_eq!(config.mode, Mode::Debug);
        assert_eq!(config.fps, 29.97);
        assert_eq!(config.camera_index, Some(3));
        assert_eq!(
            config.algorithm_config.input_resolution,
            Some(Resolution {
                width: 320,
                height: 180
            })
        );
    }

    #[test]
    fn rejects_algorithm_input_too_small_for_optical_flow() {
        let error = build_config(
            AppConfig::default(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some((1, 1)),
            None,
            None,
            AlgorithmOverrides::default(),
        )
        .unwrap_err();

        assert!(error.contains("Algorithm input"));
    }

    #[test]
    fn rejects_legacy_and_named_output_mode_together() {
        let error = build_config(
            AppConfig::default(),
            Some(true),
            None,
            None,
            None,
            Some("display".to_string()),
            None,
            None,
            None,
            None,
            None,
            AlgorithmOverrides::default(),
        )
        .unwrap_err();

        assert!(error.contains("headless and output_mode"));
    }
}
