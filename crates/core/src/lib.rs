pub mod config;
pub mod logging;
pub mod processor;
pub mod types;

use crate::config::{AppConfig, OutputMode, Resolution};
use crate::processor::{FrameProcessor, OpticalFlowProcessor, Params};
use crate::types::{Frame, ProcessingResult};
use opencv::{core, imgproc, prelude::*, videoio};
use std::time::Instant;

#[cfg(feature = "display")]
use opencv::highgui;
#[cfg(feature = "display")]
use rayon::prelude::*;

fn opencv_camera_index(index: usize) -> Result<i32, String> {
    i32::try_from(index).map_err(|_| format!("Camera index {index} exceeds the OpenCV range"))
}

fn configure_capture_resolution(
    resolution: Resolution,
    mut set: impl FnMut(i32, f64) -> opencv::Result<bool>,
) -> Result<(), String> {
    resolution.validate()?;
    for (property, value) in [
        (videoio::CAP_PROP_FRAME_WIDTH, resolution.width),
        (videoio::CAP_PROP_FRAME_HEIGHT, resolution.height),
    ] {
        if !set(property, f64::from(value)).map_err(|error| error.to_string())? {
            return Err(format!(
                "Camera rejected requested resolution {}x{}",
                resolution.width, resolution.height
            ));
        }
    }
    Ok(())
}

fn capture_fps_report(
    requested: f64,
    accepted: opencv::Result<bool>,
    reported: opencv::Result<f64>,
) -> logging::Report {
    let (actual, description) = match reported {
        Ok(value) if value.is_finite() && value > 0.0 => (Some(value), format!("{value:.2}")),
        Ok(_) => (None, "unavailable".to_string()),
        Err(error) => (None, format!("unavailable ({error})")),
    };
    let (warning, message) = match accepted {
        Ok(true) => {
            // Nominal 30/60 FPS modes commonly report 29.97/59.94.
            let mismatch = actual.is_none_or(|value| (value - requested).abs() > requested * 0.01);
            (
                mismatch,
                format!("Camera FPS: requested={requested:.2} reported={description}"),
            )
        }
        Ok(false) => (
            true,
            format!("Camera rejected FPS request {requested:.2}; reported FPS={description}"),
        ),
        Err(error) => (
            true,
            format!(
                "Camera FPS request {requested:.2} failed: {error}; reported FPS={description}"
            ),
        ),
    };
    logging::Report { warning, message }
}

#[cfg(any(feature = "display", test))]
fn resize_frame(frame: &Frame, resolution: Resolution) -> Result<Frame, String> {
    resolution.validate()?;
    let (_, height, _) = frame.validate_bgr()?;
    let raw = core::Mat::from_slice(&frame.data).map_err(|error| error.to_string())?;
    let source = raw.reshape(3, height).map_err(|error| error.to_string())?;
    let mut resized = core::Mat::default();
    let interpolation = if resolution.width < frame.width && resolution.height < frame.height {
        imgproc::INTER_AREA
    } else {
        imgproc::INTER_LINEAR
    };
    imgproc::resize(
        &source,
        &mut resized,
        core::Size::new(resolution.width as i32, resolution.height as i32),
        0.0,
        0.0,
        interpolation,
    )
    .map_err(|error| error.to_string())?;
    Ok(Frame {
        data: resized
            .data_bytes()
            .map_err(|error| error.to_string())?
            .to_vec(),
        width: resolution.width,
        height: resolution.height,
    })
}

fn normalize_captured_frame(frame: &core::Mat) -> Result<Frame, String> {
    let width = u32::try_from(frame.cols())
        .map_err(|_| format!("Captured frame width {} is invalid", frame.cols()))?;
    let height = u32::try_from(frame.rows())
        .map_err(|_| format!("Captured frame height {} is invalid", frame.rows()))?;

    if width == 0 || height == 0 {
        return Err("Captured frame dimensions must be non-zero".to_string());
    }
    if frame.depth() != core::CV_8U {
        return Err(format!(
            "Captured frame depth {} is unsupported; expected 8-bit data",
            frame.depth()
        ));
    }

    let mut bgr = core::Mat::default();
    match frame.channels() {
        1 => imgproc::cvt_color_def(frame, &mut bgr, imgproc::COLOR_GRAY2BGR),
        3 => frame.copy_to(&mut bgr),
        4 => imgproc::cvt_color_def(frame, &mut bgr, imgproc::COLOR_BGRA2BGR),
        channels => {
            return Err(format!(
                "Captured frame with {channels} channels is unsupported; expected 1, 3, or 4 channels"
            ));
        }
    }
    .map_err(|error| error.to_string())?;

    let normalized = Frame {
        data: bgr
            .data_bytes()
            .map(|data| data.to_vec())
            .map_err(|error| error.to_string())?,
        width,
        height,
    };
    normalized.validate_bgr()?;
    Ok(normalized)
}

#[cfg(feature = "display")]
fn validate_output_mode(output_mode: &OutputMode) -> Result<(), String> {
    match output_mode {
        OutputMode::Headless | OutputMode::Display => Ok(()),
        OutputMode::JsonCamera => Err("JSON camera output is not implemented".to_string()),
    }
}

#[cfg(not(feature = "display"))]
fn validate_output_mode(output_mode: &OutputMode) -> Result<(), String> {
    match output_mode {
        OutputMode::Headless => Ok(()),
        OutputMode::Display => {
            Err("Display mode requires the diff-motion-core display feature".to_string())
        }
        OutputMode::JsonCamera => Err("JSON camera output is not implemented".to_string()),
    }
}

#[cfg(feature = "display")]
#[inline]
fn blend_mask_pixels(image: &mut [u8], mask: &[u8]) {
    for (pixel, target) in image.as_chunks_mut::<3>().0.iter_mut().zip(mask) {
        if *target == 1 {
            for (channel, tint) in pixel.iter_mut().zip([0u16, 0, 255]) {
                *channel = ((u16::from(*channel) * 7 + tint * 3 + 5) / 10) as u8;
            }
        }
    }
}

#[cfg(feature = "display")]
fn blend_mask_overlay(image: &mut [u8], mask: &[u8], width: usize) {
    const MIN_PARALLEL_PIXELS: usize = 32 * 1024;

    if mask.len() < MIN_PARALLEL_PIXELS {
        blend_mask_pixels(image, mask);
        return;
    }
    image
        .par_chunks_mut(width * 3)
        .zip(mask.par_chunks(width))
        .for_each(|(image_row, mask_row)| blend_mask_pixels(image_row, mask_row));
}

#[cfg(feature = "display")]
fn prepare_display_frame(
    frame: &Frame,
    mask: &crate::types::SemanticMask,
    display_resolution: Option<Resolution>,
) -> Result<core::Mat, String> {
    let display_frame;
    let frame = if let Some(resolution) = display_resolution {
        display_frame = resize_frame(frame, resolution)?;
        &display_frame
    } else {
        frame
    };
    let (width, height, _) = frame.validate_bgr()?;
    Resolution {
        width: mask.width,
        height: mask.height,
    }
    .validate()?;
    if (mask.width as usize).checked_mul(mask.height as usize) != Some(mask.data.len()) {
        return Err("Mask dimensions do not match its buffer".to_string());
    }
    if mask.data.iter().any(|pixel| *pixel > 1) {
        return Err("Semantic mask must contain only 0 and 1".to_string());
    }
    let raw = core::Mat::from_slice(&frame.data)
        .map_err(|error| format!("Failed to prepare display frame: {error}"))?;
    let mut image = raw
        .reshape(3, height)
        .and_then(|image| image.try_clone())
        .map_err(|error| format!("Failed to copy display frame: {error}"))?;
    let mask_raw = core::Mat::from_slice(&mask.data).map_err(|error| error.to_string())?;
    let source_mask = mask_raw
        .reshape(1, mask.height as i32)
        .map_err(|error| error.to_string())?;
    let mut display_mask = core::Mat::default();
    imgproc::resize(
        &source_mask,
        &mut display_mask,
        core::Size::new(width, height),
        0.0,
        0.0,
        imgproc::INTER_NEAREST,
    )
    .map_err(|error| error.to_string())?;
    blend_mask_overlay(
        image.data_bytes_mut().map_err(|error| error.to_string())?,
        display_mask
            .data_bytes()
            .map_err(|error| error.to_string())?,
        width as usize,
    );
    Ok(image)
}

#[cfg(feature = "display")]
fn exit_requested(key: i32) -> bool {
    matches!(key & 0xff, 27 | 113 | 81)
}

#[cfg(feature = "display")]
fn window_is_closed(
    visibility: opencv::Result<f64>,
    get_rect: impl FnOnce() -> opencv::Result<core::Rect>,
) -> Result<bool, String> {
    let closed = match visibility {
        Ok(visible) if visible >= 0.0 => return Ok(visible < 1.0),
        Ok(_) => get_rect().map(|rect| rect.width <= 0 || rect.height <= 0),
        Err(error) => Err(error),
    };
    match closed {
        Ok(closed) => Ok(closed),
        Err(error) if error.code == core::StsNullPtr => Ok(true),
        Err(error) => Err(format!("Failed to inspect display window: {error}")),
    }
}

#[cfg(feature = "display")]
fn draw_fps(image: &mut core::Mat, fps: Option<f64>) -> opencv::Result<()> {
    let label = fps.map_or_else(|| "FPS: --".to_string(), |value| format!("FPS: {value:.1}"));
    for (color, thickness) in [(core::Scalar::all(0.0), 3), (core::Scalar::all(255.0), 1)] {
        imgproc::put_text(
            image,
            &label,
            core::Point::new(10, 22),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}

#[cfg(feature = "display")]
fn display_result(
    frame: &Frame,
    result: &ProcessingResult,
    window_name: &str,
    display_resolution: Option<Resolution>,
    fps: Option<f64>,
) -> Result<bool, String> {
    let mut image = prepare_display_frame(frame, &result.mask, display_resolution)?;
    draw_fps(&mut image, fps).map_err(|error| format!("Failed to render FPS: {error}"))?;
    highgui::imshow(window_name, &image)
        .map_err(|error| format!("Failed to display frame: {error}"))?;
    let key =
        highgui::wait_key(1).map_err(|error| format!("Failed to read display input: {error}"))?;
    if exit_requested(key) {
        return Ok(true);
    }
    if highgui::current_ui_framework().map_err(|error| error.to_string())? == "WAYLAND" {
        return Ok(false);
    }
    window_is_closed(
        highgui::get_window_property(window_name, highgui::WND_PROP_VISIBLE),
        || highgui::get_window_image_rect(window_name),
    )
}

pub fn process_video_stream(config: AppConfig) -> Result<(), String> {
    process_video_stream_with_callback(config, |_, _| Ok(()))
}

/// Delivers each processed frame synchronously; receiver errors stop capture.
pub fn process_video_stream_with_callback(
    config: AppConfig,
    callback: impl FnMut(&Frame, &ProcessingResult) -> Result<(), String>,
) -> Result<(), String> {
    let logger = logging::Logger::new(config.mode);
    logger.debug(format_args!(
        "Starting diff-motion with config:\n{config:#?}"
    ));
    let processing = run_video_stream(config, callback, &logger);
    if let Err(error) = &processing {
        logger.error(error);
    }
    processing
}

fn run_video_stream(
    config: AppConfig,
    mut callback: impl FnMut(&Frame, &ProcessingResult) -> Result<(), String>,
    logger: &logging::Logger,
) -> Result<(), String> {
    config.validate()?;
    validate_output_mode(&config.output_mode)?;
    let params = Params::from(&config.algorithm_config);
    logger.info("Initializing video stream processing...");

    let mut capture = if let Some(video_path) = &config.video_path {
        logger.info(format_args!("Opening video file: {video_path}"));
        videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)
            .map_err(|error| format!("Failed to open video file {video_path}: {error}"))?
    } else {
        let index = config.resolved_camera_index();
        logger.info(format_args!("Opening camera index: {index}"));
        videoio::VideoCapture::new(opencv_camera_index(index)?, videoio::CAP_ANY)
            .map_err(|error| format!("Failed to open camera {index}: {error}"))?
    };

    if !videoio::VideoCapture::is_opened(&capture)
        .map_err(|error| format!("Failed to inspect video source: {error}"))?
    {
        return Err("Failed to open video source".to_string());
    }

    if config.video_path.is_none() {
        if let Some(resolution) = config.resolution {
            configure_capture_resolution(resolution, |property, value| {
                capture.set(property, value)
            })?;
        }
        let accepted = capture.set(videoio::CAP_PROP_FPS, config.fps);
        let reported = capture.get(videoio::CAP_PROP_FPS);
        capture_fps_report(config.fps, accepted, reported).write(logger);
    }
    let mut processor = OpticalFlowProcessor::new(params)
        .map_err(|error| format!("Failed to initialize processor: {error}"))?;
    #[cfg(feature = "display")]
    let display = config.output_mode == OutputMode::Display;
    #[cfg(feature = "display")]
    let window_name = "Diff Motion Segmenter";
    #[cfg(feature = "display")]
    if display {
        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)
            .map_err(|error| format!("Failed to create display window: {error}"))?;
    }

    let mut captured = core::Mat::default();
    let mut stats = logging::StreamStats::new(Instant::now(), logger.enabled());
    let processing = (|| {
        loop {
            if !capture
                .read(&mut captured)
                .map_err(|error| format!("Failed to read frame: {error}"))?
                || captured.empty()
            {
                if config.video_path.is_none() {
                    logger.warn("Camera returned no frame; stopping capture");
                }
                break;
            }

            let frame = normalize_captured_frame(&captured)?;
            if config.video_path.is_none() {
                if let Some(resolution) = config.resolution {
                    if frame.width != resolution.width || frame.height != resolution.height {
                        return Err(format!(
                            "Camera delivered {}x{} instead of requested {}x{}",
                            frame.width, frame.height, resolution.width, resolution.height
                        ));
                    }
                }
            }
            let processing_started = Instant::now();
            let result = processor.process_frame(&frame)?;
            let processing_ms = processing_started.elapsed().as_secs_f64() * 1000.0;
            let foreground = stats.record(&result.mask, processor.ego_motion_reliable());
            logger.debug(format_args!(
                "capture={}x{} mask={}x{} processing_ms={processing_ms:.3} objects={} foreground_pixels={foreground} ego_motion_reliable={:?}",
                frame.width, frame.height, result.mask.width, result.mask.height,
                result.objects.len(), processor.ego_motion_reliable()
            ));
            callback(&frame, &result)?;

            #[cfg(feature = "display")]
            if display
                && display_result(
                    &frame,
                    &result,
                    window_name,
                    config.display_resolution,
                    stats.fps(),
                )?
            {
                break;
            }
            if let Some(report) = stats.report(Instant::now()) {
                report.write(logger);
            }
        }
        Ok(())
    })();
    stats.finish(Instant::now()).write(logger);

    #[cfg(feature = "display")]
    if display {
        let cleanup = match highgui::destroy_window(window_name) {
            Ok(()) => Ok(()),
            Err(error) if error.code == core::StsNullPtr => Ok(()),
            Err(error) => Err(format!("Failed to close display window: {error}")),
        };
        if processing.is_err() {
            if let Err(error) = cleanup {
                logger.warn(error);
            }
            return processing;
        }
        cleanup?;
        return Ok(());
    }

    processing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_fps_reporting_handles_fractional_rates_rejection_and_missing_values() {
        let fractional = capture_fps_report(30.0, Ok(true), Ok(29.97));
        assert!(!fractional.warning);
        assert!(fractional
            .message
            .contains("requested=30.00 reported=29.97"));
        assert!(capture_fps_report(60.0, Ok(true), Ok(30.0)).warning);
        let rejected = capture_fps_report(60.0, Ok(false), Ok(30.0));
        assert!(rejected.warning && rejected.message.contains("rejected"));
        for reported in [0.0, f64::NAN, f64::INFINITY] {
            let unknown = capture_fps_report(30.0, Ok(true), Ok(reported));
            assert!(unknown.warning && unknown.message.contains("unavailable"));
        }
        let failed = capture_fps_report(
            30.0,
            Err(opencv::Error::new(core::StsError, "unsupported property")),
            Ok(30.0),
        );
        assert!(failed.warning && failed.message.contains("unsupported property"));
        let failed = capture_fps_report(
            30.0,
            Ok(true),
            Err(opencv::Error::new(core::StsError, "cannot query rate")),
        );
        assert!(failed.warning && failed.message.contains("cannot query rate"));
    }

    #[cfg(feature = "display")]
    #[test]
    fn fps_label_stays_within_the_top_of_the_preview() {
        for fps in [None, Some(59.94)] {
            let mut image = core::Mat::new_rows_cols_with_default(
                128,
                256,
                core::CV_8UC3,
                core::Scalar::all(100.0),
            )
            .unwrap();
            draw_fps(&mut image, fps).unwrap();
            let pixels = image.data_bytes().unwrap();
            let top = 30 * 256 * 3;
            assert!(pixels[..top].iter().any(|&pixel| pixel != 100));
            assert!(pixels[top..].iter().all(|&pixel| pixel == 100));
        }
    }

    #[test]
    fn rejects_tiny_algorithm_input_before_opening_video() {
        let mut config = AppConfig {
            video_path: Some("missing.mp4".to_string()),
            ..AppConfig::default()
        };
        config.algorithm_config.input_resolution = Some(Resolution {
            width: 1,
            height: 1,
        });
        let error = process_video_stream(config).unwrap_err();
        assert!(error.contains("Algorithm input"));
    }

    #[cfg(feature = "display")]
    #[test]
    fn scales_binary_mask_to_camera_overlay_with_nearest_neighbour() {
        let frame = Frame {
            width: 4,
            height: 2,
            data: vec![100; 24],
        };
        let mask = crate::types::SemanticMask {
            width: 2,
            height: 1,
            data: vec![1, 0],
        };
        let image = prepare_display_frame(&frame, &mask, None).unwrap();
        assert_eq!(
            image.data_bytes().unwrap(),
            &[
                70, 70, 147, 70, 70, 147, 100, 100, 100, 100, 100, 100, 70, 70, 147, 70, 70, 147,
                100, 100, 100, 100, 100, 100,
            ]
        );
    }

    #[cfg(feature = "display")]
    #[test]
    fn resizes_the_final_display_frame() {
        let frame = Frame {
            width: 4,
            height: 2,
            data: vec![100; 24],
        };
        let mask = crate::types::SemanticMask {
            width: 2,
            height: 1,
            data: vec![1, 0],
        };

        let image = prepare_display_frame(
            &frame,
            &mask,
            Some(Resolution {
                width: 2,
                height: 1,
            }),
        )
        .unwrap();

        assert_eq!(image.size().unwrap(), core::Size::new(2, 1));
        assert_eq!(image.data_bytes().unwrap(), &[70, 70, 147, 100, 100, 100]);
    }

    #[test]
    fn resizes_algorithm_input_in_both_directions() {
        let frame = Frame {
            width: 4,
            height: 2,
            data: vec![100; 24],
        };
        for resolution in [
            Resolution {
                width: 2,
                height: 1,
            },
            Resolution {
                width: 8,
                height: 4,
            },
        ] {
            let resized = resize_frame(&frame, resolution).unwrap();
            assert_eq!(
                (resized.width, resized.height),
                (resolution.width, resolution.height)
            );
            resized.validate_bgr().unwrap();
            assert!(resized.data.iter().all(|value| *value == 100));
        }
        assert_eq!(frame.data, vec![100; 24]);
    }

    #[test]
    fn requests_camera_resolution_and_reports_driver_rejection() {
        let resolution = crate::config::Resolution {
            width: 640,
            height: 480,
        };
        let mut properties = Vec::new();
        configure_capture_resolution(resolution, |property, value| {
            properties.push((property, value));
            Ok(true)
        })
        .unwrap();
        assert_eq!(
            properties,
            vec![
                (videoio::CAP_PROP_FRAME_WIDTH, 640.0),
                (videoio::CAP_PROP_FRAME_HEIGHT, 480.0),
            ]
        );
        assert!(configure_capture_resolution(resolution, |_, _| Ok(false)).is_err());
        let error = configure_capture_resolution(resolution, |_, _| {
            Err(opencv::Error::new(core::StsError, "camera disconnected"))
        })
        .unwrap_err();
        assert!(error.contains("camera disconnected"));
    }

    #[cfg(feature = "display")]
    #[test]
    fn overlays_only_target_pixels_without_changing_input() {
        let frame = Frame {
            width: 5,
            height: 4,
            data: (0..20).flat_map(|i| [40 + i, 80 + i, 120 + i]).collect(),
        };
        let mask = crate::types::SemanticMask {
            width: 5,
            height: 4,
            data: vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        let image = prepare_display_frame(&frame, &mask, None).unwrap();
        assert_eq!((image.cols(), image.rows(), image.channels()), (5, 4, 3));
        for (index, pixel) in image
            .data_bytes()
            .unwrap()
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
        {
            let source = &frame.data[index * 3..index * 3 + 3];
            if mask.data[index] == 0 {
                assert_eq!(pixel, source);
            } else {
                assert!(pixel[0] < source[0]);
                assert!(pixel[1] < source[1]);
                assert!(pixel[2] > source[2]);
            }
        }
        assert_eq!(&image.data_bytes().unwrap()[18..21], &[32, 60, 165]);
        assert_eq!(&frame.data[18..21], &[46, 86, 126]);
        assert_eq!(mask.data[6], 1);
    }

    #[cfg(feature = "display")]
    #[test]
    fn overlay_is_identical_with_one_and_four_rayon_threads() {
        let width = 640usize;
        let height = 360usize;
        let source: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        ((x * 3 + y) % 256) as u8,
                        ((x + y * 5) % 256) as u8,
                        ((x * 7 + y * 11) % 256) as u8,
                    ]
                })
            })
            .collect();
        let mask: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width)
                    .map(move |x| u8::from((300..340).contains(&x) && (166..194).contains(&y)))
            })
            .collect();
        let render = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut image = source.clone();
                blend_mask_overlay(&mut image, &mask, width);
                image
            })
        };

        assert_eq!(render(1), render(4));
    }

    #[cfg(feature = "display")]
    #[test]
    fn renders_empty_and_full_masks_at_channel_boundaries() {
        let frame = Frame {
            width: 2,
            height: 1,
            data: vec![0, 255, 255, 255, 0, 0],
        };
        for (data, expected) in [
            (vec![0, 0], frame.data.clone()),
            (vec![1, 1], vec![0, 179, 255, 179, 0, 77]),
        ] {
            let mask = crate::types::SemanticMask {
                width: 2,
                height: 1,
                data,
            };
            let image = prepare_display_frame(&frame, &mask, None).unwrap();
            assert_eq!(image.data_bytes().unwrap(), expected);
        }
    }

    #[cfg(feature = "display")]
    #[test]
    fn rejects_invalid_display_masks() {
        let frame = Frame {
            width: 2,
            height: 2,
            data: vec![80; 12],
        };
        for (width, height, data) in [
            (0, 1, vec![]),
            (2, 2, vec![0; 3]),
            (2, 2, vec![2; 4]),
            (1, 4, vec![0; 3]),
        ] {
            let mask = crate::types::SemanticMask {
                width,
                height,
                data,
            };
            assert!(prepare_display_frame(&frame, &mask, None).is_err());
        }
    }

    #[cfg(feature = "display")]
    #[test]
    fn exits_display_on_q_or_escape() {
        for key in [27, i32::from(b'q'), i32::from(b'Q')] {
            assert!(exit_requested(key));
        }
        for key in [-1, i32::from(b'a')] {
            assert!(!exit_requested(key));
        }
    }

    #[cfg(feature = "display")]
    #[test]
    fn detects_closed_windows_without_hiding_backend_errors() {
        let visible_rect = || Ok(core::Rect::new(0, 0, 640, 480));
        assert!(!window_is_closed(Ok(1.0), visible_rect).unwrap());
        assert!(window_is_closed(Ok(0.0), visible_rect).unwrap());
        assert!(!window_is_closed(Ok(-1.0), visible_rect).unwrap());
        assert!(window_is_closed(Ok(-1.0), || {
            Err(opencv::Error::new(core::StsNullPtr, "NULL window"))
        })
        .unwrap());
        assert!(window_is_closed(
            Err(opencv::Error::new(core::StsNullPtr, "NULL window")),
            visible_rect,
        )
        .unwrap());
        assert!(window_is_closed(Ok(-1.0), || {
            Err(opencv::Error::new(
                core::StsNotImplemented,
                "Unsupported backend",
            ))
        })
        .is_err());
    }

    #[test]
    fn delivers_headless_frame_and_stops_on_receiver_error() {
        let config = AppConfig {
            video_path: Some(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../fixtures/gemini_generated_video_8813b1b0.mp4")
                    .to_string_lossy()
                    .into_owned(),
            ),
            ..AppConfig::default()
        };
        let expected = config.algorithm_config.resolved_input_resolution();
        let mut received = 0;
        let error = process_video_stream_with_callback(config, |frame, result| {
            received += 1;
            assert_eq!(
                (result.mask.width, result.mask.height),
                (expected.width, expected.height)
            );
            assert_eq!(
                result.mask.data.len(),
                expected.width as usize * expected.height as usize
            );
            assert!(frame.width > 0 && frame.height > 0);
            assert!(result.mask.data.iter().all(|value| *value == 0));
            assert!(result.objects.is_empty());
            Err("receiver stopped".to_string())
        })
        .unwrap_err();
        assert_eq!(received, 1);
        assert_eq!(error, "receiver stopped");
    }

    #[test]
    fn rejects_camera_index_outside_opencv_range() {
        assert!(opencv_camera_index(usize::MAX).is_err());
    }

    #[test]
    fn accepts_camera_index_in_opencv_range() {
        assert_eq!(opencv_camera_index(2).unwrap(), 2);
    }

    #[test]
    fn normalizes_grayscale_frame_to_bgr() {
        let frame =
            core::Mat::new_rows_cols_with_default(1, 2, core::CV_8UC1, core::Scalar::all(7.0))
                .unwrap();

        let normalized = normalize_captured_frame(&frame).unwrap();

        assert_eq!((normalized.width, normalized.height), (2, 1));
        assert_eq!(normalized.data, vec![7, 7, 7, 7, 7, 7]);
    }

    #[test]
    fn copies_bgr_frame_into_owned_contiguous_buffer() {
        let frame = core::Mat::new_rows_cols_with_default(
            1,
            1,
            core::CV_8UC3,
            core::Scalar::new(1.0, 2.0, 3.0, 0.0),
        )
        .unwrap();

        let normalized = normalize_captured_frame(&frame).unwrap();

        assert_eq!((normalized.width, normalized.height), (1, 1));
        assert_eq!(normalized.data, vec![1, 2, 3]);
    }

    #[test]
    fn normalizes_bgra_frame_to_bgr() {
        let frame = core::Mat::new_rows_cols_with_default(
            1,
            1,
            core::CV_8UC4,
            core::Scalar::new(1.0, 2.0, 3.0, 4.0),
        )
        .unwrap();

        let normalized = normalize_captured_frame(&frame).unwrap();

        assert_eq!((normalized.width, normalized.height), (1, 1));
        assert_eq!(normalized.data, vec![1, 2, 3]);
    }

    #[test]
    fn rejects_unsupported_captured_frame_channels() {
        let frame =
            core::Mat::new_rows_cols_with_default(1, 1, core::CV_8UC2, core::Scalar::all(0.0))
                .unwrap();

        let error = normalize_captured_frame(&frame).unwrap_err();

        assert!(error.contains("2 channels"));
    }

    #[test]
    fn accepts_headless_mode_in_every_build() {
        assert!(validate_output_mode(&OutputMode::Headless).is_ok());
    }

    #[cfg(feature = "display")]
    #[test]
    fn accepts_display_mode_when_display_support_is_enabled() {
        assert!(validate_output_mode(&OutputMode::Display).is_ok());
    }

    #[cfg(not(feature = "display"))]
    #[test]
    fn rejects_display_mode_when_display_support_is_disabled() {
        let error = validate_output_mode(&OutputMode::Display).unwrap_err();

        assert!(error.contains("display feature"));
    }

    #[test]
    fn rejects_unimplemented_json_output() {
        let error = validate_output_mode(&OutputMode::JsonCamera).unwrap_err();

        assert!(error.contains("not implemented"));
    }
}
