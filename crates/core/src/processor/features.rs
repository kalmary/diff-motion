use opencv::{core, video};

#[derive(Default)]
pub(super) struct FeaturePyramids {
    pub(super) previous: core::Vector<core::Mat>,
    pub(super) current: core::Vector<core::Mat>,
}

impl FeaturePyramids {
    pub(super) fn prepare(
        &mut self,
        previous: &core::Mat,
        current: &core::Mat,
    ) -> opencv::Result<()> {
        if self.previous.is_empty() {
            video::build_optical_flow_pyramid_def(
                previous,
                &mut self.previous,
                core::Size::new(21, 21),
                3,
            )?;
        } else {
            std::mem::swap(&mut self.previous, &mut self.current);
        }

        video::build_optical_flow_pyramid_def(
            current,
            &mut self.current,
            core::Size::new(21, 21),
            3,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::{features, imgproc, prelude::*};

    struct FlowResult {
        points: core::Vector<core::Point2f>,
        status: core::Vector<u8>,
        errors: core::Vector<f32>,
    }

    fn textured_frame() -> opencv::Result<core::Mat> {
        let mut frame =
            core::Mat::new_rows_cols_with_default(240, 320, core::CV_8UC1, core::Scalar::all(0.0))?;
        for y in 0..frame.rows() {
            for x in 0..frame.cols() {
                let value = ((x * 31 + y * 17 + (x * y) % 251) % 256) as u8;
                *frame.at_2d_mut::<u8>(y, x)? = value;
            }
        }
        Ok(frame)
    }

    fn translated_frame(
        source: &core::Mat,
        horizontal: f64,
        vertical: f64,
    ) -> opencv::Result<core::Mat> {
        let transform = core::Mat::from_slice_2d(&[[1.0, 0.0, horizontal], [0.0, 1.0, vertical]])?;
        let mut result = core::Mat::default();
        imgproc::warp_affine_def(source, &mut result, &transform, source.size()?)?;
        Ok(result)
    }

    fn tracked_points(frame: &core::Mat) -> opencv::Result<core::Vector<core::Point2f>> {
        let mut points = core::Vector::new();
        features::good_features_to_track(
            frame,
            &mut points,
            200,
            0.01,
            8.0,
            &core::no_array(),
            3,
            false,
            0.04,
        )?;
        Ok(points)
    }

    fn calculate_flow(
        previous: &impl core::ToInputArray,
        current: &impl core::ToInputArray,
        points: &core::Vector<core::Point2f>,
    ) -> opencv::Result<FlowResult> {
        let mut result = FlowResult {
            points: core::Vector::new(),
            status: core::Vector::new(),
            errors: core::Vector::new(),
        };
        video::calc_optical_flow_pyr_lk(
            previous,
            current,
            points,
            &mut result.points,
            &mut result.status,
            &mut result.errors,
            core::Size::new(21, 21),
            3,
            core::TermCriteria::new(core::TermCriteria_COUNT + core::TermCriteria_EPS, 20, 0.03)?,
            0,
            1e-4,
        )?;
        Ok(result)
    }

    fn assert_same_flow(expected: &FlowResult, actual: &FlowResult) -> opencv::Result<()> {
        assert_eq!(actual.points.len(), expected.points.len());
        assert_eq!(actual.status.to_vec(), expected.status.to_vec());
        for index in 0..expected.points.len() {
            let expected_point = expected.points.get(index)?;
            let actual_point = actual.points.get(index)?;
            assert_eq!(actual_point.x.to_bits(), expected_point.x.to_bits());
            assert_eq!(actual_point.y.to_bits(), expected_point.y.to_bits());
            if expected.status.get(index)? == 1 {
                assert_eq!(
                    actual.errors.get(index)?.to_bits(),
                    expected.errors.get(index)?.to_bits()
                );
            }
        }
        Ok(())
    }

    fn assert_pyramid_matches_images(
        previous: &core::Mat,
        current: &core::Mat,
        pyramids: &FeaturePyramids,
    ) -> opencv::Result<()> {
        let points = tracked_points(previous)?;
        assert!(!points.is_empty());

        let raw_forward = calculate_flow(previous, current, &points)?;
        let pyramid_forward = calculate_flow(&pyramids.previous, &pyramids.current, &points)?;
        assert_same_flow(&raw_forward, &pyramid_forward)?;

        let raw_backward = calculate_flow(current, previous, &raw_forward.points)?;
        let pyramid_backward = calculate_flow(
            &pyramids.current,
            &pyramids.previous,
            &pyramid_forward.points,
        )?;
        assert_same_flow(&raw_backward, &pyramid_backward)
    }

    #[test]
    fn reused_pyramids_match_raw_lk_across_frames() -> opencv::Result<()> {
        let first = textured_frame()?;
        let second = translated_frame(&first, 2.0, -1.0)?;
        let third = translated_frame(&first, 4.0, -2.0)?;
        let mut pyramids = FeaturePyramids::default();

        pyramids.prepare(&first, &second)?;
        assert_pyramid_matches_images(&first, &second, &pyramids)?;

        pyramids.prepare(&second, &third)?;
        assert_pyramid_matches_images(&second, &third, &pyramids)
    }
}
