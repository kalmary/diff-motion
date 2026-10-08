pub mod config;
pub mod types;
pub mod processor;

use opencv::{videoio, highgui, imgproc, core, prelude::*};
use crate::config::{AppConfig, OutputMode};
use crate::processor::{OpticalFlowProcessor, Params, FrameProcessor};
use crate::types::Frame;

pub fn process_video_stream(config: AppConfig) {
    println!("Initializing video stream processing...");

    let mut cap = if let Some(video_path) = &config.video_path {
        println!("Opening video file: {}", video_path);
        videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY).expect("Failed to open video file")
    } else {
        let idx = config.camera_index.unwrap_or(0);
        println!("Opening camera index: {}", idx);
        videoio::VideoCapture::new(idx as i32, videoio::CAP_ANY).expect("Failed to open camera")
    };

    if !videoio::VideoCapture::is_opened(&cap).unwrap_or(false) {
        panic!("Failed to open video source!");
    }

    let mut processor = OpticalFlowProcessor::new(Params::default()).expect("Failed to init processor");

    let display = config.output_mode == OutputMode::Display;
    let window_name = "Diff Motion Segmenter";
    if display {
        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE).unwrap();
    }

    let mut frame_mat = core::Mat::default();

    loop {
        if !cap.read(&mut frame_mat).unwrap_or(false) || frame_mat.empty() {
            println!("End of stream or failed to read frame");
            break;
        }

        let width = frame_mat.cols() as u32;
        let height = frame_mat.rows() as u32;
        let total_bytes = (frame_mat.total() * frame_mat.elem_size().unwrap()) as usize;
        let mut data = vec![0u8; total_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(frame_mat.data(), data.as_mut_ptr(), total_bytes);
        }

        let frame = Frame { data, width, height };

        let result = processor.process_frame(&frame).unwrap();

        if display {
            // Visualize results
            let mut display_mat = frame_mat.clone();

            // Overlay mask
            let mask = result.mask;
            let mask_raw = core::Mat::from_slice(&mask.data).unwrap();
            let mask_mat_raw = mask_raw.reshape(1, height as i32).unwrap();
            
            let mut mask_mat = core::Mat::default();
            mask_mat_raw.copy_to(&mut mask_mat).unwrap();

            let mut red_mask = core::Mat::default();
            let zeros = core::Mat::new_rows_cols_with_default(height as i32, width as i32, core::CV_8UC1, core::Scalar::all(0.0)).unwrap();
            let mut channels = core::Vector::<core::Mat>::new();
            channels.push(zeros.clone());
            channels.push(zeros.clone());
            channels.push(mask_mat); // Red channel
            core::merge(&channels, &mut red_mask).unwrap();

            let display_mat_cloned = display_mat.clone();
            core::add_weighted(&display_mat_cloned, 0.7, &red_mask, 0.3, 0.0, &mut display_mat, -1).unwrap();

            // Draw bounding boxes
            for obj in result.objects {
                let rect = core::Rect::new(
                    obj.bbox.x as i32,
                    obj.bbox.y as i32,
                    obj.bbox.width as i32,
                    obj.bbox.height as i32,
                );
                imgproc::rectangle(&mut display_mat, rect, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0).unwrap();
            }

            highgui::imshow(window_name, &display_mat).unwrap();

            if highgui::wait_key(1).unwrap() == 27 { // ESC
                break;
            }
        }
    }
}
