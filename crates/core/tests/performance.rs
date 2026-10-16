use std::{collections::hash_map::DefaultHasher, hash::Hasher, time::Instant};

use diff_motion_core::{
    processor::{FrameProcessor, OpticalFlowProcessor, Params},
    types::Frame,
};
use opencv::{core, prelude::*, videoio};

#[test]
#[ignore = "release throughput benchmark; run alone with --nocapture"]
fn benchmark_processing_throughput() -> opencv::Result<()> {
    if let Ok(threads) = std::env::var("DIFF_MOTION_BENCH_OPENCV_THREADS") {
        core::set_num_threads(threads.parse().expect("thread count must be an integer"))?;
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/gemini_generated_video_8813b1b0.mp4");
    let mut capture = videoio::VideoCapture::from_file(path.to_str().unwrap(), videoio::CAP_ANY)?;
    let mut frames = Vec::new();
    let mut decoded = core::Mat::default();
    while frames.len() < 120 && capture.read(&mut decoded)? && !decoded.empty() {
        frames.push(Frame {
            data: decoded.data_bytes()?.to_vec(),
            width: decoded.cols() as u32,
            height: decoded.rows() as u32,
        });
    }
    assert!(frames.len() > 1, "benchmark requires decoded video frames");
    let params = Params::default();
    println!(
        "input={}x{}, processing={}x{}, OpenCV threads={}, Rayon threads={}",
        frames[0].width,
        frames[0].height,
        params.w,
        params.h,
        core::get_num_threads()?,
        rayon::current_num_threads()
    );
    let mut reference_fingerprint = None;
    for repeat in 0..3 {
        let mut processor = OpticalFlowProcessor::new(Params::default())?;
        let mut fingerprint = DefaultHasher::new();
        let mut times = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let started = Instant::now();
            let result = processor
                .process_frame(frame)
                .map_err(|error| opencv::Error::new(core::StsError, error))?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            if index > 0 {
                times.push(elapsed);
            }
            fingerprint.write(&result.mask.data);
            for object in result.objects {
                for value in [
                    object.bbox.x,
                    object.bbox.y,
                    object.bbox.width,
                    object.bbox.height,
                    object.confidence,
                ] {
                    fingerprint.write_u32(value.to_bits());
                }
            }
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        times.sort_by(f64::total_cmp);
        let fingerprint = fingerprint.finish();
        if let Some(reference) = reference_fingerprint {
            assert_eq!(
                fingerprint, reference,
                "processing output changed between runs"
            );
        }
        reference_fingerprint = Some(fingerprint);
        println!(
            "run {repeat}: mean={mean:.3}ms p50={:.3}ms p95={:.3}ms throughput={:.1}FPS fingerprint={:016x}",
            times[times.len() / 2],
            times[times.len() * 95 / 100],
            1000.0 / mean,
            fingerprint
        );
    }
    Ok(())
}
