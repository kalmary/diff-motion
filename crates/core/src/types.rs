#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
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
