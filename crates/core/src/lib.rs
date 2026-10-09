pub mod config;
pub mod processor;
pub mod types;

use crate::config::{AppConfig, OutputMode};
use crate::processor::{FrameProcessor, OpticalFlowProcessor, Params};
use crate::types::Frame;
use opencv::{core, imgproc, prelude::*, videoio};

#[cfg(feature = "display")]
use crate::types::ProcessingResult;
#[cfg(feature = "display")]
use opencv::highgui;

fn opencv_camera_index(index: usize) -> Result<i32, String> {
    i32::try_from(index).map_err(|_| format!("Camera index {index} exceeds the OpenCV range"))
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
fn display_result(
    frame: &Frame,
    result: ProcessingResult,
    window_name: &str,
) -> Result<bool, String> {
    let (width, height, _) = frame.validate_bgr()?;
    let display_raw = core::Mat::from_slice(&frame.data)
        .map_err(|error| format!("Failed to prepare display frame: {error}"))?;
    let display_reshaped = display_raw
        .reshape(3, height)
        .map_err(|error| format!("Failed to reshape display frame: {error}"))?;
    let mut display_mat = core::Mat::default();
    display_reshaped
        .copy_to(&mut display_mat)
        .map_err(|error| format!("Failed to copy display frame: {error}"))?;

    let mask_raw = core::Mat::from_slice(&result.mask.data)
        .map_err(|error| format!("Failed to prepare display mask: {error}"))?;
    let mask_reshaped = mask_raw
        .reshape(1, height)
        .map_err(|error| format!("Failed to reshape display mask: {error}"))?;
    let mut mask = core::Mat::default();
    mask_reshaped
        .copy_to(&mut mask)
        .map_err(|error| format!("Failed to copy display mask: {error}"))?;

    let zeros =
        core::Mat::new_rows_cols_with_default(height, width, core::CV_8UC1, core::Scalar::all(0.0))
            .map_err(|error| format!("Failed to allocate display overlay: {error}"))?;
    let mut channels = core::Vector::<core::Mat>::new();
    channels.push(zeros.clone());
    channels.push(zeros);
    channels.push(mask);
    let mut red_mask = core::Mat::default();
    core::merge(&channels, &mut red_mask)
        .map_err(|error| format!("Failed to create display overlay: {error}"))?;

    let source = display_mat.clone();
    core::add_weighted(&source, 0.7, &red_mask, 0.3, 0.0, &mut display_mat, -1)
        .map_err(|error| format!("Failed to blend display overlay: {error}"))?;

    for object in result.objects {
        let rectangle = core::Rect::new(
            object.bbox.x as i32,
            object.bbox.y as i32,
            object.bbox.width as i32,
            object.bbox.height as i32,
        );
        imgproc::rectangle(
            &mut display_mat,
            rectangle,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )
        .map_err(|error| format!("Failed to draw bounding box: {error}"))?;
    }

    highgui::imshow(window_name, &display_mat)
        .map_err(|error| format!("Failed to display frame: {error}"))?;
    highgui::wait_key(1)
        .map(|key| key == 27)
        .map_err(|error| format!("Failed to read display input: {error}"))
}

pub fn process_video_stream(config: AppConfig) -> Result<(), String> {
    validate_output_mode(&config.output_mode)?;
    println!("Initializing video stream processing...");

    let mut capture = if let Some(video_path) = &config.video_path {
        println!("Opening video file: {video_path}");
        videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)
            .map_err(|error| format!("Failed to open video file {video_path}: {error}"))?
    } else {
        let index = config.camera_index.unwrap_or(0);
        println!("Opening camera index: {index}");
        videoio::VideoCapture::new(opencv_camera_index(index)?, videoio::CAP_ANY)
            .map_err(|error| format!("Failed to open camera {index}: {error}"))?
    };

    if !videoio::VideoCapture::is_opened(&capture)
        .map_err(|error| format!("Failed to inspect video source: {error}"))?
    {
        return Err("Failed to open video source".to_string());
    }

    let mut processor = OpticalFlowProcessor::new(Params::default())
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
    loop {
        if !capture
            .read(&mut captured)
            .map_err(|error| format!("Failed to read frame: {error}"))?
            || captured.empty()
        {
            break;
        }

        let frame = normalize_captured_frame(&captured)?;
        let result = processor.process_frame(&frame)?;

        #[cfg(feature = "display")]
        if display && display_result(&frame, result, window_name)? {
            break;
        }

        #[cfg(not(feature = "display"))]
        let _ = result;
    }

    #[cfg(feature = "display")]
    if display {
        highgui::destroy_window(window_name)
            .map_err(|error| format!("Failed to close display window: {error}"))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
