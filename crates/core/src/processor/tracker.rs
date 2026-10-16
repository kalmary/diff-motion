use std::collections::VecDeque;

use crate::types::{BoundingBox, DetectedObject};
use opencv::{core, imgproc, prelude::*, video, Result as CvResult};

#[derive(Clone, Copy, PartialEq)]
enum TrackState {
    Search,
    Lock,
    Coast,
}

pub(super) struct Tracker {
    state: TrackState,
    kalman: video::KalmanFilter,
    bbox: Option<core::Rect2f>,
    frame_size: core::Size,
    template: Option<core::Mat>,
    last_mask: Option<core::Mat>,
    last_object: Option<DetectedObject>,
    homography: [f64; 9],
    coast_frames: u8,
    candidate: Option<DetectedObject>,
    history: VecDeque<bool>,
    confirmation_window: usize,
    confirmation_hits: usize,
    gate_scale: f32,
    max_coast_frames: u8,
    min_correlation: f64,
}

impl Tracker {
    pub(super) fn new(params: &crate::processor::Params) -> CvResult<Self> {
        let mut kalman = video::KalmanFilter::new(4, 2, 0, core::CV_32F)?;

        let mut transition = matrix(4, 4)?;
        for i in 0..4 {
            *transition.at_2d_mut::<f32>(i, i)? = 1.0;
        }
        *transition.at_2d_mut::<f32>(0, 2)? = 1.0;
        *transition.at_2d_mut::<f32>(1, 3)? = 1.0;
        kalman.set_transition_matrix(transition);

        let mut measurement = matrix(2, 4)?;
        *measurement.at_2d_mut::<f32>(0, 0)? = 1.0;
        *measurement.at_2d_mut::<f32>(1, 1)? = 1.0;
        kalman.set_measurement_matrix(measurement);
        kalman.set_process_noise_cov(diagonal(4, 1e-2)?);
        kalman.set_measurement_noise_cov(diagonal(2, 1e-1)?);
        kalman.set_error_cov_post(diagonal(4, 1.0)?);

        Ok(Self {
            state: TrackState::Search,
            kalman,
            bbox: None,
            frame_size: core::Size::new(0, 0),
            template: None,
            last_mask: None,
            last_object: None,
            homography: identity_homography(),
            coast_frames: 0,
            candidate: None,
            history: VecDeque::new(),
            confirmation_window: params.confirmation_window,
            confirmation_hits: params.confirmation_hits,
            gate_scale: params.tracker_gate_scale,
            max_coast_frames: params.max_coast_frames,
            min_correlation: params.min_correlation,
        })
    }

    pub(super) fn predict(&mut self, h: &[f64; 9], size: core::Size) -> CvResult<()> {
        self.frame_size = size;
        self.homography = *h;
        if self.state == TrackState::Search {
            return Ok(());
        }

        let Some(bbox) = self.bbox else {
            self.reset();
            return Ok(());
        };
        if size.width <= 0 || size.height <= 0 {
            self.reset();
            return Ok(());
        }

        let state = self.kalman.state_post();
        let center = core::Point2f::new(*state.at_2d::<f32>(0, 0)?, *state.at_2d::<f32>(1, 0)?);
        let velocity_end = core::Point2f::new(
            center.x + *state.at_2d::<f32>(2, 0)?,
            center.y + *state.at_2d::<f32>(3, 0)?,
        );
        let Some(aligned_center) = transform_point(center, h) else {
            self.reset();
            return Ok(());
        };
        let Some(aligned_end) = transform_point(velocity_end, h) else {
            self.reset();
            return Ok(());
        };

        let mut aligned_state = matrix(4, 1)?;
        *aligned_state.at_2d_mut::<f32>(0, 0)? = aligned_center.x;
        *aligned_state.at_2d_mut::<f32>(1, 0)? = aligned_center.y;
        *aligned_state.at_2d_mut::<f32>(2, 0)? = aligned_end.x - aligned_center.x;
        *aligned_state.at_2d_mut::<f32>(3, 0)? = aligned_end.y - aligned_center.y;
        kalman_state(&mut self.kalman, &aligned_state)?;

        let predicted = self.kalman.predict_def()?;
        let x = *predicted.at_2d::<f32>(0, 0)?;
        let y = *predicted.at_2d::<f32>(1, 0)?;
        if !x.is_finite() || !y.is_finite() {
            self.reset();
            return Ok(());
        }
        self.bbox = Some(core::Rect2f::new(
            x - bbox.width / 2.0,
            y - bbox.height / 2.0,
            bbox.width,
            bbox.height,
        ));
        if self.gate().is_none() {
            self.reset();
        }
        Ok(())
    }

    pub(super) fn gate(&self) -> Option<core::Rect> {
        if self.state == TrackState::Search {
            return None;
        }
        let bbox = self.bbox?;
        let margin = (self.gate_scale - 1.0) / 2.0;
        let expanded = core::Rect2f::new(
            bbox.x - bbox.width * margin,
            bbox.y - bbox.height * margin,
            bbox.width * self.gate_scale,
            bbox.height * self.gate_scale,
        );
        clipped_rect(expanded, self.frame_size)
    }

    pub(super) fn feature_exclusion(&self) -> Option<core::Rect> {
        self.bbox
            .and_then(|bbox| clipped_rect(bbox, self.frame_size))
    }

    pub(super) fn update(
        &mut self,
        curr: &core::Mat,
        mask: &mut core::Mat,
        object: Option<DetectedObject>,
        ego_ok: bool,
    ) -> CvResult<Option<DetectedObject>> {
        validate_images(curr, mask)?;
        self.frame_size = curr.size()?;

        let object = match object {
            Some(object) if object_rect(&object, self.frame_size).is_some() => Some(object),
            Some(_) => {
                self.reset();
                clear_mask(mask)?;
                return Ok(None);
            }
            None => None,
        };

        match self.state {
            TrackState::Search => self.search(curr, mask, object, ego_ok),
            TrackState::Lock | TrackState::Coast => {
                if ego_ok {
                    if let Some(object) = object {
                        if self.associated(&object) && self.appearance_matches(curr, &object)? {
                            return self.lock(curr, mask, object).map(Some);
                        }
                    }
                }
                self.coast(curr, mask)
            }
        }
    }

    fn search(
        &mut self,
        curr: &core::Mat,
        mask: &mut core::Mat,
        object: Option<DetectedObject>,
        ego_ok: bool,
    ) -> CvResult<Option<DetectedObject>> {
        let Some(object) = object else {
            if ego_ok && self.candidate.is_some() {
                self.push_history(false);
            }
            return Ok(None);
        };
        if !ego_ok {
            clear_mask(mask)?;
            return Ok(None);
        }

        let consistent = self
            .candidate
            .as_ref()
            .is_some_and(|candidate| candidate_matches(candidate, &object));
        if consistent {
            self.candidate = Some(object.clone());
            self.push_history(true);
        } else {
            self.candidate = Some(object.clone());
            self.history.clear();
            self.push_history(true);
        }

        if self.history.iter().filter(|&&matched| matched).count() >= self.confirmation_hits {
            self.initialize_filter(&object)?;
            self.state = TrackState::Lock;
            self.candidate = None;
            self.history.clear();
            self.save_lock_data(curr, mask, &object)?;
            return Ok(Some(object));
        }
        clear_mask(mask)?;
        Ok(None)
    }

    fn lock(
        &mut self,
        curr: &core::Mat,
        mask: &core::Mat,
        object: DetectedObject,
    ) -> CvResult<DetectedObject> {
        let rect = tracked_rect(&object, self.frame_size)?;
        let corrected = self.correct(rect)?;
        self.bbox = Some(core::Rect2f::new(
            corrected.x - rect.width as f32 / 2.0,
            corrected.y - rect.height as f32 / 2.0,
            rect.width as f32,
            rect.height as f32,
        ));
        self.state = TrackState::Lock;
        self.coast_frames = 0;
        self.save_lock_data(curr, mask, &object)?;
        Ok(object)
    }

    fn coast(
        &mut self,
        curr: &core::Mat,
        mask: &mut core::Mat,
    ) -> CvResult<Option<DetectedObject>> {
        if self.coast_frames >= self.max_coast_frames {
            self.reset();
            clear_mask(mask)?;
            return Ok(None);
        }
        let (Some(gate), Some(template), Some(mut object)) = (
            self.gate(),
            self.template.as_ref(),
            self.last_object.clone(),
        ) else {
            self.reset();
            clear_mask(mask)?;
            return Ok(None);
        };
        if gate.width < template.cols() || gate.height < template.rows() {
            self.reset();
            clear_mask(mask)?;
            return Ok(None);
        }

        let search = curr.roi(gate)?;
        if !has_texture(&search)? {
            self.reset();
            clear_mask(mask)?;
            return Ok(None);
        }
        let mut result = core::Mat::default();
        imgproc::match_template_def(&search, template, &mut result, imgproc::TM_CCOEFF_NORMED)?;
        let mut correlation = 0.0;
        let mut location = core::Point::default();
        core::min_max_loc(
            &result,
            None,
            Some(&mut correlation),
            None,
            Some(&mut location),
            &core::no_array(),
        )?;
        if !correlation.is_finite() || correlation < self.min_correlation {
            self.reset();
            clear_mask(mask)?;
            return Ok(None);
        }

        let matched = core::Rect::new(
            gate.x + location.x,
            gate.y + location.y,
            template.cols(),
            template.rows(),
        );
        let center = self.correct(matched)?;
        self.bbox = Some(core::Rect2f::new(
            center.x - matched.width as f32 / 2.0,
            center.y - matched.height as f32 / 2.0,
            matched.width as f32,
            matched.height as f32,
        ));
        self.restore_mask(mask, &object, matched)?;
        self.state = TrackState::Coast;
        self.coast_frames += 1;

        object.bbox = BoundingBox {
            x: matched.x as f32,
            y: matched.y as f32,
            width: matched.width as f32,
            height: matched.height as f32,
        };
        self.last_object = Some(object.clone());
        Ok(Some(object))
    }

    fn initialize_filter(&mut self, object: &DetectedObject) -> CvResult<()> {
        let rect = tracked_rect(object, self.frame_size)?;
        let center = rect_center(rect);
        let mut state = matrix(4, 1)?;
        *state.at_2d_mut::<f32>(0, 0)? = center.x;
        *state.at_2d_mut::<f32>(1, 0)? = center.y;
        kalman_state(&mut self.kalman, &state)?;
        self.kalman.set_error_cov_post(diagonal(4, 1.0)?);
        self.bbox = Some(core::Rect2f::new(
            rect.x as f32,
            rect.y as f32,
            rect.width as f32,
            rect.height as f32,
        ));
        Ok(())
    }

    fn correct(&mut self, rect: core::Rect) -> CvResult<core::Point2f> {
        let center = rect_center(rect);
        let mut measurement = matrix(2, 1)?;
        *measurement.at_2d_mut::<f32>(0, 0)? = center.x;
        *measurement.at_2d_mut::<f32>(1, 0)? = center.y;
        let corrected = self.kalman.correct(&measurement)?;
        Ok(core::Point2f::new(
            *corrected.at_2d::<f32>(0, 0)?,
            *corrected.at_2d::<f32>(1, 0)?,
        ))
    }

    fn associated(&self, object: &DetectedObject) -> bool {
        let (Some(gate), Some(rect)) = (self.gate(), object_rect(object, self.frame_size)) else {
            return false;
        };
        let center = rect_center(rect);
        center.x >= gate.x as f32
            && center.y >= gate.y as f32
            && center.x < (gate.x + gate.width) as f32
            && center.y < (gate.y + gate.height) as f32
    }

    fn appearance_matches(&self, curr: &core::Mat, object: &DetectedObject) -> CvResult<bool> {
        let Some(template) = self.template.as_ref() else {
            return Ok(true);
        };
        let Some(rect) = object_rect(object, self.frame_size) else {
            return Ok(false);
        };
        let crop = curr.roi(rect)?;
        if !has_texture(&crop)? {
            return Ok(false);
        }

        let mut resized = core::Mat::default();
        imgproc::resize(
            &crop,
            &mut resized,
            template.size()?,
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        let mut result = core::Mat::default();
        imgproc::match_template_def(
            &resized,
            template,
            &mut result,
            imgproc::TM_CCOEFF_NORMED,
        )?;
        let mut correlation = 0.0;
        core::min_max_loc(
            &result,
            None,
            Some(&mut correlation),
            None,
            None,
            &core::no_array(),
        )?;
        Ok(correlation.is_finite() && correlation >= self.min_correlation)
    }

    fn save_lock_data(
        &mut self,
        curr: &core::Mat,
        mask: &core::Mat,
        object: &DetectedObject,
    ) -> CvResult<()> {
        let rect = tracked_rect(object, self.frame_size)?;
        let crop = curr.roi(rect)?;
        if has_texture(&crop)? {
            let mut template = core::Mat::default();
            crop.copy_to(&mut template)?;
            self.template = Some(template);
        } else {
            self.template = None;
        }
        self.last_mask = Some(copy_mat(mask)?);
        self.last_object = Some(object.clone());
        Ok(())
    }

    fn restore_mask(
        &mut self,
        mask: &mut core::Mat,
        previous_object: &DetectedObject,
        matched: core::Rect,
    ) -> CvResult<()> {
        let Some(previous_mask) = self.last_mask.as_ref() else {
            clear_mask(mask)?;
            return Ok(());
        };
        let previous = tracked_rect(previous_object, self.frame_size)?;
        let Some(aligned_center) = transform_point(rect_center(previous), &self.homography) else {
            clear_mask(mask)?;
            return Ok(());
        };
        let matched_center = rect_center(matched);
        let dx = f64::from(matched_center.x - aligned_center.x);
        let dy = f64::from(matched_center.y - aligned_center.y);
        let mut transform = self.homography;
        transform[0] += dx * transform[6];
        transform[1] += dx * transform[7];
        transform[2] += dx * transform[8];
        transform[3] += dy * transform[6];
        transform[4] += dy * transform[7];
        transform[5] += dy * transform[8];
        let h_mat = homography_mat(&transform)?;
        imgproc::warp_perspective(
            previous_mask,
            mask,
            &h_mat,
            self.frame_size,
            imgproc::INTER_NEAREST,
            core::BORDER_CONSTANT,
            core::Scalar::all(0.0),
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        self.last_mask = Some(copy_mat(mask)?);
        Ok(())
    }

    fn push_history(&mut self, matched: bool) {
        if self.history.len() == self.confirmation_window {
            self.history.pop_front();
        }
        self.history.push_back(matched);
    }

    fn reset(&mut self) {
        self.state = TrackState::Search;
        self.bbox = None;
        self.template = None;
        self.last_mask = None;
        self.last_object = None;
        self.coast_frames = 0;
        self.candidate = None;
        self.history.clear();
    }
}

fn validate_images(curr: &core::Mat, mask: &core::Mat) -> CvResult<()> {
    if curr.empty()
        || curr.typ() != core::CV_8UC1
        || mask.typ() != core::CV_8UC1
        || curr.size()? != mask.size()?
    {
        return Err(opencv::Error::new(
            core::StsBadArg,
            "Tracker expects equally sized CV_8UC1 image and mask",
        ));
    }
    Ok(())
}

fn candidate_matches(previous: &DetectedObject, current: &DetectedObject) -> bool {
    let previous_center = core::Point2f::new(
        previous.bbox.x + previous.bbox.width / 2.0,
        previous.bbox.y + previous.bbox.height / 2.0,
    );
    let current_center = core::Point2f::new(
        current.bbox.x + current.bbox.width / 2.0,
        current.bbox.y + current.bbox.height / 2.0,
    );
    let limit = previous
        .bbox
        .width
        .max(previous.bbox.height)
        .max(current.bbox.width.max(current.bbox.height));
    (current_center.x - previous_center.x).abs() <= limit
        && (current_center.y - previous_center.y).abs() <= limit
}

fn object_rect(object: &DetectedObject, size: core::Size) -> Option<core::Rect> {
    let bbox = &object.bbox;
    if size.width <= 0
        || size.height <= 0
        || !bbox.x.is_finite()
        || !bbox.y.is_finite()
        || !bbox.width.is_finite()
        || !bbox.height.is_finite()
        || bbox.width < 1.0
        || bbox.height < 1.0
    {
        return None;
    }
    clipped_rect(
        core::Rect2f::new(bbox.x, bbox.y, bbox.width, bbox.height),
        size,
    )
}

fn tracked_rect(object: &DetectedObject, size: core::Size) -> CvResult<core::Rect> {
    object_rect(object, size).ok_or_else(|| {
        opencv::Error::new(
            core::StsBadArg,
            "Tracked object has an invalid bounding box",
        )
    })
}

fn clipped_rect(rect: core::Rect2f, size: core::Size) -> Option<core::Rect> {
    if !rect.x.is_finite()
        || !rect.y.is_finite()
        || !rect.width.is_finite()
        || !rect.height.is_finite()
        || rect.width <= 0.0
        || rect.height <= 0.0
    {
        return None;
    }
    let left = rect.x.floor().max(0.0).min(size.width as f32) as i32;
    let top = rect.y.floor().max(0.0).min(size.height as f32) as i32;
    let right = (rect.x + rect.width).ceil().max(0.0).min(size.width as f32) as i32;
    let bottom = (rect.y + rect.height)
        .ceil()
        .max(0.0)
        .min(size.height as f32) as i32;
    (right > left && bottom > top).then(|| core::Rect::new(left, top, right - left, bottom - top))
}

fn transform_point(point: core::Point2f, h: &[f64; 9]) -> Option<core::Point2f> {
    if !h.iter().all(|value| value.is_finite()) {
        return None;
    }
    let x = point.x as f64;
    let y = point.y as f64;
    let scale = h[6] * x + h[7] * y + h[8];
    if !scale.is_finite() || scale.abs() < 1e-9 {
        return None;
    }
    let mapped_x = (h[0] * x + h[1] * y + h[2]) / scale;
    let mapped_y = (h[3] * x + h[4] * y + h[5]) / scale;
    (mapped_x.is_finite() && mapped_y.is_finite())
        .then(|| core::Point2f::new(mapped_x as f32, mapped_y as f32))
}

fn rect_center(rect: core::Rect) -> core::Point2f {
    core::Point2f::new(
        rect.x as f32 + rect.width as f32 / 2.0,
        rect.y as f32 + rect.height as f32 / 2.0,
    )
}

fn matrix(rows: i32, cols: i32) -> CvResult<core::Mat> {
    core::Mat::new_rows_cols_with_default(rows, cols, core::CV_32F, core::Scalar::all(0.0))
}

fn diagonal(size: i32, value: f32) -> CvResult<core::Mat> {
    let mut result = matrix(size, size)?;
    for i in 0..size {
        *result.at_2d_mut::<f32>(i, i)? = value;
    }
    Ok(result)
}

fn kalman_state(kalman: &mut video::KalmanFilter, state: &core::Mat) -> CvResult<()> {
    kalman.set_state_post(copy_mat(state)?);
    kalman.set_state_pre(copy_mat(state)?);
    Ok(())
}

fn has_texture(image: &impl core::ToInputArray) -> CvResult<bool> {
    let mut mean = core::Scalar::default();
    let mut deviation = core::Scalar::default();
    core::mean_std_dev(image, &mut mean, &mut deviation, &core::no_array())?;
    Ok(deviation[0].is_finite() && deviation[0] > 1e-6)
}

fn homography_mat(h: &[f64; 9]) -> CvResult<core::Mat> {
    let mut result =
        core::Mat::new_rows_cols_with_default(3, 3, core::CV_64F, core::Scalar::all(0.0))?;
    for row in 0..3 {
        for col in 0..3 {
            *result.at_2d_mut::<f64>(row, col)? = h[(row * 3 + col) as usize];
        }
    }
    Ok(result)
}

fn copy_mat(source: &impl core::MatTraitConst) -> CvResult<core::Mat> {
    let mut result = core::Mat::default();
    source.copy_to(&mut result)?;
    Ok(result)
}

fn clear_mask(mask: &mut core::Mat) -> CvResult<()> {
    mask.set_to(&core::Scalar::all(0.0), &core::no_array())?;
    Ok(())
}

fn identity_homography() -> [f64; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: core::Size = core::Size::new(48, 36);

    #[test]
    fn locks_only_after_three_consistent_good_ego_frames() -> CvResult<()> {
        let mut tracker = Tracker::new(&crate::processor::Params::default())?;
        let frame = textured_frame(core::Rect::new(12, 10, 8, 8))?;

        for _ in 0..3 {
            let mut mask = object_mask(core::Rect::new(12, 10, 8, 8))?;
            assert!(tracker
                .update(&frame, &mut mask, Some(object(12.0, 10.0)), false)?
                .is_none());
            assert_eq!(core::count_non_zero(&mask)?, 0);
        }
        assert!(tracker.gate().is_none());

        for index in 0..3 {
            let mut mask = object_mask(core::Rect::new(12, 10, 8, 8))?;
            let tracked = tracker.update(&frame, &mut mask, Some(object(12.0, 10.0)), true)?;
            if index < 2 {
                assert!(tracked.is_none());
                assert_eq!(core::count_non_zero(&mask)?, 0);
            } else {
                assert!(tracked.is_some());
                assert_eq!(core::count_non_zero(&mask)?, 64);
            }
        }
        assert!(tracker.gate().is_some());
        Ok(())
    }

    #[test]
    fn disabled_coasting_keeps_observations_and_clears_missing_masks() -> CvResult<()> {
        let params = crate::processor::Params {
            confirmation_window: 1,
            confirmation_hits: 1,
            max_coast_frames: 0,
            ..crate::processor::Params::default()
        };
        let mut tracker = Tracker::new(&params)?;
        let rect = core::Rect::new(12, 10, 8, 8);
        let frame = textured_frame(rect)?;
        let mut mask = object_mask(rect)?;
        tracker.update(&frame, &mut mask, Some(object(12.0, 10.0)), true)?;

        let moved = core::Rect::new(16, 10, 8, 8);
        let frame = textured_frame(moved)?;
        tracker.predict(&identity_homography(), SIZE)?;
        mask = object_mask(moved)?;
        let observed = mask.data_bytes()?.to_vec();
        let result = tracker.update(&frame, &mut mask, Some(object(16.0, 10.0)), true)?;
        assert_eq!(result.unwrap().bbox.x, 16.0);
        assert_eq!(mask.data_bytes()?, observed);

        tracker.predict(&identity_homography(), SIZE)?;
        assert!(tracker.update(&frame, &mut mask, None, true)?.is_none());
        assert_eq!(core::count_non_zero(&mask)?, 0);
        assert!(tracker.gate().is_none());
        Ok(())
    }

    #[test]
    fn coasts_with_texture_and_expires_after_fifteen_frames() -> CvResult<()> {
        let mut tracker = locked_tracker()?;
        let max_coast_frames = tracker.max_coast_frames;
        let current_rect = core::Rect::new(16, 10, 8, 8);
        let frame = textured_frame(current_rect)?;
        let translated = [1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let identity = identity_homography();

        for index in 0..max_coast_frames {
            tracker.predict(if index == 0 { &translated } else { &identity }, SIZE)?;
            let mut mask = object_mask(core::Rect::new(12, 10, 8, 8))?;
            let tracked = tracker.update(&frame, &mut mask, None, true)?;
            assert!(tracked.is_some());
            assert_eq!(mask_overlap(&mask, current_rect)?, 64);
        }

        tracker.predict(&identity_homography(), SIZE)?;
        let mut mask = object_mask(core::Rect::new(12, 10, 8, 8))?;
        assert!(tracker.update(&frame, &mut mask, None, true)?.is_none());
        assert_eq!(core::count_non_zero(&mask)?, 0);
        assert!(tracker.gate().is_none());
        Ok(())
    }

    #[test]
    fn rejects_unrelated_motion_and_drops_a_disappeared_target() -> CvResult<()> {
        let mut tracker = locked_tracker()?;
        tracker.predict(&identity_homography(), SIZE)?;
        let blank = core::Mat::new_rows_cols_with_default(
            SIZE.height,
            SIZE.width,
            core::CV_8UC1,
            core::Scalar::all(0.0),
        )?;
        let mut mask = object_mask(core::Rect::new(36, 26, 8, 8))?;

        assert!(tracker
            .update(&blank, &mut mask, Some(object(36.0, 26.0)), true)?
            .is_none());
        assert_eq!(core::count_non_zero(&mask)?, 0);
        assert!(tracker.gate().is_none());
        Ok(())
    }

    #[test]
    fn unreliable_ego_coasts_instead_of_refreshing_from_motion() -> CvResult<()> {
        let mut tracker = locked_tracker()?;
        tracker.predict(&identity_homography(), SIZE)?;
        let target = core::Rect::new(12, 10, 8, 8);
        let frame = textured_frame(target)?;
        let unrelated = core::Rect::new(4, 2, 24, 24);
        let mut mask = object_mask(unrelated)?;
        let unrelated_object = DetectedObject {
            bbox: BoundingBox {
                x: 4.0,
                y: 2.0,
                width: 24.0,
                height: 24.0,
            },
            ..object(4.0, 2.0)
        };

        let tracked = tracker
            .update(&frame, &mut mask, Some(unrelated_object), false)?
            .expect("the existing target should coast through unreliable ego motion");

        assert_eq!((tracked.bbox.x, tracked.bbox.y), (12.0, 10.0));
        assert_eq!((tracked.bbox.width, tracked.bbox.height), (8.0, 8.0));
        assert_eq!(mask_overlap(&mask, target)?, 64);
        assert_eq!(core::count_non_zero(&mask)?, 64);
        Ok(())
    }

    #[test]
    fn different_appearance_inside_gate_coasts_on_the_existing_target() -> CvResult<()> {
        let mut tracker = locked_tracker()?;
        tracker.predict(&identity_homography(), SIZE)?;
        let target = core::Rect::new(12, 10, 8, 8);
        let distractor = core::Rect::new(20, 10, 8, 8);
        let mut frame = textured_frame(target)?;
        for y in distractor.y..distractor.y + distractor.height {
            for x in distractor.x..distractor.x + distractor.width {
                *frame.at_2d_mut::<u8>(y, x)? = if x % 2 == 0 { 40 } else { 220 };
            }
        }
        let mut mask = object_mask(distractor)?;

        let tracked = tracker
            .update(&frame, &mut mask, Some(object(20.0, 10.0)), true)?
            .expect("the existing target should be retained");

        assert_eq!((tracked.bbox.x, tracked.bbox.y), (12.0, 10.0));
        assert_eq!(mask_overlap(&mask, target)?, 64);
        assert_eq!(core::count_non_zero(&mask)?, 64);
        Ok(())
    }

    #[test]
    fn feature_exclusion_uses_the_tracked_box_instead_of_the_gate() -> CvResult<()> {
        let tracker = locked_tracker()?;

        assert_eq!(tracker.gate(), Some(core::Rect::new(4, 2, 24, 24)));
        assert_eq!(
            tracker.feature_exclusion(),
            Some(core::Rect::new(12, 10, 8, 8))
        );
        Ok(())
    }

    #[test]
    fn confirmation_and_gate_settings_change_tracker_behavior() -> CvResult<()> {
        let params = crate::processor::Params {
            confirmation_window: 2,
            confirmation_hits: 2,
            tracker_gate_scale: 1.0,
            ..crate::processor::Params::default()
        };
        let mut tracker = Tracker::new(&params)?;
        let frame = textured_frame(core::Rect::new(12, 10, 8, 8))?;

        for _ in 0..2 {
            let mut mask = object_mask(core::Rect::new(12, 10, 8, 8))?;
            tracker.update(&frame, &mut mask, Some(object(12.0, 10.0)), true)?;
        }

        assert_eq!(tracker.gate().unwrap().size(), core::Size::new(8, 8));
        Ok(())
    }

    fn locked_tracker() -> CvResult<Tracker> {
        let mut tracker = Tracker::new(&crate::processor::Params::default())?;
        let frame = textured_frame(core::Rect::new(12, 10, 8, 8))?;
        for _ in 0..3 {
            let mut mask = object_mask(core::Rect::new(12, 10, 8, 8))?;
            tracker.update(&frame, &mut mask, Some(object(12.0, 10.0)), true)?;
        }
        Ok(tracker)
    }

    fn object(x: f32, y: f32) -> DetectedObject {
        DetectedObject {
            class_id: 1,
            class_name: "UAV".to_string(),
            confidence: 1.0,
            bbox: BoundingBox {
                x,
                y,
                width: 8.0,
                height: 8.0,
            },
        }
    }

    fn textured_frame(rect: core::Rect) -> CvResult<core::Mat> {
        let mut frame = core::Mat::new_rows_cols_with_default(
            SIZE.height,
            SIZE.width,
            core::CV_8UC1,
            core::Scalar::all(0.0),
        )?;
        for y in rect.y..rect.y + rect.height {
            for x in rect.x..rect.x + rect.width {
                *frame.at_2d_mut::<u8>(y, x)? = if (x + y) % 2 == 0 { 40 } else { 220 };
            }
        }
        Ok(frame)
    }

    fn object_mask(rect: core::Rect) -> CvResult<core::Mat> {
        let mut mask = core::Mat::new_rows_cols_with_default(
            SIZE.height,
            SIZE.width,
            core::CV_8UC1,
            core::Scalar::all(0.0),
        )?;
        imgproc::rectangle(
            &mut mask,
            rect,
            core::Scalar::all(1.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        Ok(mask)
    }

    fn mask_overlap(mask: &core::Mat, rect: core::Rect) -> CvResult<i32> {
        let crop = mask.roi(rect)?;
        core::count_non_zero(&crop)
    }
}
