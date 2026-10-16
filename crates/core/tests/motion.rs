use diff_motion_core::{
    processor::{FrameProcessor, OpticalFlowProcessor, Params},
    types::{BoundingBox, Frame, ProcessingResult},
};
use opencv::{core, imgproc, prelude::*, videoio};

const WIDTH: i32 = 640;
const HEIGHT: i32 = 360;
const TARGET_WIDTH: i32 = 40;
const TARGET_HEIGHT: i32 = 28;

fn params() -> Params {
    Params {
        w: WIDTH,
        h: HEIGHT,
        ..Params::default()
    }
}

fn load_background() -> opencv::Result<core::Mat> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/gemini_generated_video_8813b1b0.mp4");
    let mut capture = videoio::VideoCapture::from_file(
        path.to_str().expect("fixture path must be valid UTF-8"),
        videoio::CAP_ANY,
    )?;
    let mut decoded = core::Mat::default();
    assert!(capture.read(&mut decoded)? && !decoded.empty());

    let mut background = core::Mat::default();
    imgproc::resize(
        &decoded,
        &mut background,
        core::Size::new(WIDTH, HEIGHT),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    Ok(background)
}

fn target_patch(background: &core::Mat) -> opencv::Result<core::Mat> {
    let source = core::Mat::roi(
        background,
        core::Rect::new(280, 300, TARGET_WIDTH, TARGET_HEIGHT),
    )?;
    let mut patch = core::Mat::default();
    source.copy_to(&mut patch)?;
    Ok(patch)
}

fn translated_frame(
    background: &core::Mat,
    offset: (i32, i32),
    target: Option<(&core::Mat, core::Rect)>,
) -> opencv::Result<Frame> {
    let transform = core::Mat::from_slice_2d(&[
        [1.0, 0.0, f64::from(offset.0)],
        [0.0, 1.0, f64::from(offset.1)],
    ])?;
    let mut image = core::Mat::default();
    imgproc::warp_affine_def(
        background,
        &mut image,
        &transform,
        core::Size::new(WIDTH, HEIGHT),
    )?;

    if let Some((patch, rect)) = target {
        let mut destination = core::Mat::roi_mut(&mut image, rect)?;
        patch.copy_to(&mut destination)?;
    }

    Ok(Frame {
        data: image.data_bytes()?.to_vec(),
        width: WIDTH as u32,
        height: HEIGHT as u32,
    })
}

fn assert_mask_contract(result: &ProcessingResult) {
    assert_eq!(
        (result.mask.width, result.mask.height),
        (WIDTH as u32, HEIGHT as u32)
    );
    assert_eq!(result.mask.data.len(), (WIDTH * HEIGHT) as usize);
    assert!(result.mask.data.iter().all(|value| *value <= 1));
}

fn bbox_iou(actual: &BoundingBox, expected: core::Rect) -> f32 {
    let left = actual.x.max(expected.x as f32);
    let top = actual.y.max(expected.y as f32);
    let right = (actual.x + actual.width).min((expected.x + expected.width) as f32);
    let bottom = (actual.y + actual.height).min((expected.y + expected.height) as f32);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let union =
        actual.width * actual.height + (expected.width * expected.height) as f32 - intersection;
    intersection / union
}

fn mask_iou(result: &ProcessingResult, expected: core::Rect) -> f32 {
    let mut predicted = 0usize;
    let mut intersection = 0usize;
    for (index, value) in result.mask.data.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        predicted += 1;
        let x = index as i32 % WIDTH;
        let y = index as i32 / WIDTH;
        if x >= expected.x
            && x < expected.x + expected.width
            && y >= expected.y
            && y < expected.y + expected.height
        {
            intersection += 1;
        }
    }
    let expected_area = (expected.width * expected.height) as usize;
    intersection as f32 / (predicted + expected_area - intersection) as f32
}

fn assert_localized(result: &ProcessingResult, expected: core::Rect, phase: &str, index: usize) {
    assert_mask_contract(result);
    let object = result
        .objects
        .first()
        .expect("target must produce a bounding box");
    let box_overlap = bbox_iou(&object.bbox, expected);
    let mask_overlap = mask_iou(result, expected);
    println!("{phase} frame {index}: bbox IoU={box_overlap:.3}, mask IoU={mask_overlap:.3}");
    assert!(
        box_overlap >= 0.5,
        "bounding-box IoU {box_overlap:.3} is below 0.5; bbox={:?}, target={expected:?}",
        object.bbox
    );
    assert!(
        mask_overlap >= 0.5,
        "mask IoU {mask_overlap:.3} is below 0.5 for target {expected:?}"
    );
}

#[test]
fn global_translation_and_jitter_keep_false_mask_area_low() -> opencv::Result<()> {
    let background = load_background()?;
    let offsets = [(0, 0), (2, 1), (5, 0), (7, 2), (10, 1), (12, 3)];
    let mut processor = OpticalFlowProcessor::new(params())?;
    let mut max_foreground = 0;

    for offset in offsets {
        let frame = translated_frame(&background, offset, None)?;
        let result = processor
            .process_frame(&frame)
            .map_err(|message| opencv::Error::new(core::StsError, message))?;
        assert_mask_contract(&result);
        let foreground = result.mask.data.iter().filter(|value| **value == 1).count();
        max_foreground = max_foreground.max(foreground);
        assert!(
            foreground <= (TARGET_WIDTH * TARGET_HEIGHT) as usize / 10,
            "false mask area was {:.3}% ({foreground} pixels), above 10% of target area: {:?}",
            foreground as f64 * 100.0 / result.mask.data.len() as f64,
            result.objects
        );
    }
    println!(
        "background maximum: {max_foreground} foreground pixels ({:.3}% of frame)",
        max_foreground as f64 * 100.0 / f64::from(WIDTH * HEIGHT)
    );
    Ok(())
}

#[test]
fn moving_patch_is_segmented_and_localized() -> opencv::Result<()> {
    let background = load_background()?;
    let patch = target_patch(&background)?;
    let offsets = [(0, 0), (2, 1), (5, 0), (7, 2), (10, 1), (12, 3)];
    let mut processor = OpticalFlowProcessor::new(params())?;
    let mut last = None;

    for (index, offset) in offsets.into_iter().enumerate() {
        let target = core::Rect::new(
            145 + offset.0 + index as i32 * 8,
            215 + offset.1,
            TARGET_WIDTH,
            TARGET_HEIGHT,
        );
        let frame = translated_frame(&background, offset, Some((&patch, target)))?;
        let result = processor
            .process_frame(&frame)
            .map_err(|message| opencv::Error::new(core::StsError, message))?;
        last = Some((result, target));
    }

    let (result, target) = last.expect("sequence must contain frames");
    assert_localized(&result, target, "moving", offsets.len() - 1);
    Ok(())
}

#[test]
fn abrupt_target_reversal_keeps_mask_on_current_position() -> opencv::Result<()> {
    let background = load_background()?;
    let patch = target_patch(&background)?;
    let horizontal_motion = [0, 8, 16, 24, 32, 40, 24, 8];
    let mut processor = OpticalFlowProcessor::new(params())?;

    for (index, motion) in horizontal_motion.into_iter().enumerate() {
        let target = core::Rect::new(145 + motion, 215, TARGET_WIDTH, TARGET_HEIGHT);
        let frame = translated_frame(&background, (0, 0), Some((&patch, target)))?;
        let result = processor
            .process_frame(&frame)
            .map_err(|message| opencv::Error::new(core::StsError, message))?;

        if index >= 6 {
            assert_localized(&result, target, "reversal", index);
            assert!(
                mask_iou(&result, target) >= 0.65,
                "reversed target mask remained near its previous position"
            );
        }
    }
    Ok(())
}

#[test]
fn stable_target_does_not_expand_into_background() -> opencv::Result<()> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/gemini_generated_video_8813b1b0.mp4");
    let mut capture = videoio::VideoCapture::from_file(path.to_str().unwrap(), videoio::CAP_ANY)?;
    let mut processor = OpticalFlowProcessor::new(Params {
        w: 256,
        h: 256,
        max_coast_frames: 120,
        ..Params::default()
    })?;
    let mut image = core::Mat::default();
    for index in 1..=130 {
        assert!(capture.read(&mut image)? && !image.empty());
        let frame = Frame {
            data: image.data_bytes()?.to_vec(),
            width: image.cols() as u32,
            height: image.rows() as u32,
        };
        let result = processor.process_frame(&frame).unwrap();
        if index % 10 == 0 {
            println!("fixture frame {index}: {:?}, foreground={}", result.objects, result.mask.data.iter().filter(|&&pixel| pixel == 1).count());
        }
        let target_region = match index {
            100 => core::Rect::new(75, 55, 120, 125),
            130 => core::Rect::new(45, 65, 180, 90),
            _ => continue,
        };
        let mut inside = 0;
        let mut outside = 0;
        for (pixel, &value) in result.mask.data.iter().enumerate() {
            if value == 0 {
                continue;
            }
            if target_region.contains(core::Point::new(pixel as i32 % 256, pixel as i32 / 256)) {
                inside += 1;
            } else {
                outside += 1;
            }
        }
        assert!(inside >= 12, "frame {index}: target mask disappeared");
        assert!(
            outside <= inside / 4,
            "frame {index}: background takeover: inside={inside}, outside={outside}"
        );
    }
    Ok(())
}

#[test]
fn track_coasts_when_target_motion_temporarily_matches_background() -> opencv::Result<()> {
    let background = load_background()?;
    let patch = target_patch(&background)?;
    let offsets = [
        (0, 0),
        (2, 1),
        (5, 0),
        (7, 2),
        (10, 1),
        (12, 3),
        (15, 2),
        (17, 4),
        (20, 3),
        (22, 5),
        (25, 4),
    ];
    let independent_motion = [0, 8, 16, 24, 32, 40, 40, 40, 40, 48, 56];
    let mut processor = OpticalFlowProcessor::new(params())?;

    for (index, (offset, motion)) in offsets.into_iter().zip(independent_motion).enumerate() {
        let target = core::Rect::new(
            145 + offset.0 + motion,
            215 + offset.1,
            TARGET_WIDTH,
            TARGET_HEIGHT,
        );
        let frame = translated_frame(&background, offset, Some((&patch, target)))?;
        let result = processor
            .process_frame(&frame)
            .map_err(|message| opencv::Error::new(core::StsError, message))?;

        if index >= 3 {
            let phase = match index {
                3..=5 => "acquire",
                6..=8 => "coast",
                _ => "resume",
            };
            assert_localized(&result, target, phase, index);
        }
    }
    Ok(())
}
