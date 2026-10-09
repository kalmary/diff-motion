#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

fn checked_bgr_len(width: usize, height: usize) -> Result<usize, String> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "Frame dimensions overflow the BGR buffer size".to_string())
}

impl Frame {
    pub fn validate_bgr(&self) -> Result<(i32, i32, usize), String> {
        if self.width == 0 || self.height == 0 {
            return Err("Frame dimensions must be non-zero".to_string());
        }

        let width = i32::try_from(self.width)
            .map_err(|_| format!("Frame width {} exceeds the OpenCV range", self.width))?;
        let height = i32::try_from(self.height)
            .map_err(|_| format!("Frame height {} exceeds the OpenCV range", self.height))?;
        let width_usize = usize::try_from(self.width)
            .map_err(|_| format!("Frame width {} exceeds the platform range", self.width))?;
        let height_usize = usize::try_from(self.height)
            .map_err(|_| format!("Frame height {} exceeds the platform range", self.height))?;
        let expected_len = checked_bgr_len(width_usize, height_usize)?;

        if self.data.len() != expected_len {
            return Err(format!(
                "BGR frame buffer length mismatch: expected {expected_len} bytes, got {}",
                self.data.len()
            ));
        }

        Ok((width, height, width_usize * height_usize))
    }
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone)]
pub struct SemanticMask {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub objects: Vec<DetectedObject>,
    pub mask: SemanticMask,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, data: Vec<u8>) -> Frame {
        Frame {
            data,
            width,
            height,
        }
    }

    #[test]
    fn rejects_zero_frame_dimensions() {
        assert!(frame(0, 1, Vec::new()).validate_bgr().is_err());
        assert!(frame(1, 0, Vec::new()).validate_bgr().is_err());
    }

    #[test]
    fn rejects_frame_buffer_length_mismatch() {
        let error = frame(2, 2, vec![0; 11]).validate_bgr().unwrap_err();

        assert!(error.contains("12"));
        assert!(error.contains("11"));
    }

    #[test]
    fn rejects_dimensions_outside_opencv_range() {
        let error = frame(u32::MAX, 1, Vec::new()).validate_bgr().unwrap_err();

        assert!(error.contains("OpenCV"));
    }

    #[test]
    fn rejects_bgr_buffer_size_overflow() {
        assert!(checked_bgr_len(usize::MAX, 1).is_err());
    }

    #[test]
    fn accepts_valid_bgr_frame() {
        assert!(frame(2, 2, vec![0; 12]).validate_bgr().is_ok());
    }

    #[test]
    fn test_types_instantiation() {
        let bbox = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };

        let obj = DetectedObject {
            class_id: 1,
            class_name: "UAV".to_string(),
            confidence: 0.95,
            bbox,
        };

        let mask = SemanticMask {
            data: vec![1, 0, 1, 0],
            width: 2,
            height: 2,
        };

        let result = ProcessingResult {
            objects: vec![obj],
            mask,
        };

        assert_eq!(result.objects.len(), 1);
        assert_eq!(result.objects[0].class_name, "UAV");
        assert_eq!(result.mask.width, 2);
    }
}
