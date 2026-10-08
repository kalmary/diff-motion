use crate::types::{Frame, ProcessingResult};

pub trait FrameProcessor {
    /// Process a single frame and return the detected objects and semantic mask.
    fn process_frame(&mut self, frame: &Frame) -> Result<ProcessingResult, String>;
}

pub struct DummyProcessor;

impl DummyProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl FrameProcessor for DummyProcessor {
    fn process_frame(&mut self, frame: &Frame) -> Result<ProcessingResult, String> {
        // Return an empty result for now
        Ok(ProcessingResult {
            objects: vec![],
            mask: crate::types::SemanticMask {
                data: vec![0; (frame.width * frame.height) as usize],
                width: frame.width,
                height: frame.height,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Frame;

    #[test]
    fn test_dummy_processor_returns_empty_result() {
        let mut processor = DummyProcessor::new();
        let frame = Frame {
            data: vec![0; 100 * 100 * 3],
            width: 100,
            height: 100,
        };

        let result = processor.process_frame(&frame).unwrap();

        assert!(result.objects.is_empty());
        assert_eq!(result.mask.width, 100);
        assert_eq!(result.mask.height, 100);
        assert_eq!(result.mask.data.len(), 10000);
        assert!(result.mask.data.iter().all(|&pixel| pixel == 0));
    }
}
