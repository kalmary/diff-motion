use crate::{
    config::{AlgorithmConfig, FlowPreset, Resolution},
    types::{Frame, ProcessingResult},
};
use opencv::{core, geometry, imgproc, prelude::*, video, Result as CvResult};
use rayon::prelude::*;

mod tracker;
use tracker::Tracker;
mod features;
use features::FeaturePyramids;

pub struct Params {
    pub w: i32,
    pub h: i32,
    pub fb_max: f32,
    pub ransac_h: f64,
    pub tau_min: f32,
    pub k_sigma: f32,
    pub tau_epi: f32,
    pub max_corners: i32,
    pub quality_level: f64,
    pub min_distance: f64,
    pub min_inliers: usize,
    pub parallax_threshold: f64,
    pub flow_preset: FlowPreset,
    pub finest_scale: i32,
    pub morphology_open_size: i32,
    pub morphology_close_size: i32,
    pub min_component_area: i32,
    pub search_score_threshold: f32,
    pub tracking_score_threshold: f32,
    pub confirmation_window: usize,
    pub confirmation_hits: usize,
    pub tracker_gate_scale: f32,
    /// Maximum template-propagated frames; zero disables mask propagation.
    pub max_coast_frames: u8,
    /// Appearance agreement required for motion observations and template coasting.
    pub min_correlation: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self::from(&AlgorithmConfig::default())
    }
}

impl Params {
    fn validate(&self) -> CvResult<()> {
        let width = u32::try_from(self.w)
            .map_err(|_| opencv::Error::new(core::StsBadArg, "Invalid processing width"))?;
        let height = u32::try_from(self.h)
            .map_err(|_| opencv::Error::new(core::StsBadArg, "Invalid processing height"))?;
        AlgorithmConfig {
            processing_type: "motion".to_string(),
            input_resolution: Some(Resolution { width, height }),
            fb_max: self.fb_max,
            ransac_h: self.ransac_h,
            tau_min: self.tau_min,
            k_sigma: self.k_sigma,
            tau_epi: self.tau_epi,
            max_corners: self.max_corners,
            quality_level: self.quality_level,
            min_distance: self.min_distance,
            min_inliers: self.min_inliers,
            parallax_threshold: self.parallax_threshold,
            flow_preset: self.flow_preset,
            finest_scale: self.finest_scale,
            morphology_open_size: self.morphology_open_size,
            morphology_close_size: self.morphology_close_size,
            min_component_area: self.min_component_area,
            search_score_threshold: self.search_score_threshold,
            tracking_score_threshold: self.tracking_score_threshold,
            confirmation_window: self.confirmation_window,
            confirmation_hits: self.confirmation_hits,
            tracker_gate_scale: self.tracker_gate_scale,
            max_coast_frames: self.max_coast_frames,
            min_correlation: self.min_correlation,
        }
        .validate()
        .map_err(|error| opencv::Error::new(core::StsBadArg, error))
    }
}

impl From<&AlgorithmConfig> for Params {
    fn from(config: &AlgorithmConfig) -> Self {
        let resolution = config.resolved_input_resolution();
        Self {
            w: resolution.width as i32,
            h: resolution.height as i32,
            fb_max: config.fb_max,
            ransac_h: config.ransac_h,
            tau_min: config.tau_min,
            k_sigma: config.k_sigma,
            tau_epi: config.tau_epi,
            max_corners: config.max_corners,
            quality_level: config.quality_level,
            min_distance: config.min_distance,
            min_inliers: config.min_inliers,
            parallax_threshold: config.parallax_threshold,
            flow_preset: config.flow_preset,
            finest_scale: config.finest_scale,
            morphology_open_size: config.morphology_open_size,
            morphology_close_size: config.morphology_close_size,
            min_component_area: config.min_component_area,
            search_score_threshold: config.search_score_threshold,
            tracking_score_threshold: config.tracking_score_threshold,
            confirmation_window: config.confirmation_window,
            confirmation_hits: config.confirmation_hits,
            tracker_gate_scale: config.tracker_gate_scale,
            max_coast_frames: config.max_coast_frames,
            min_correlation: config.min_correlation,
        }
    }
}

pub struct Ws {
    pub prev: core::Mat,
    pub curr: core::Mat,
    pub prev_w: core::Mat,
    pub valid: core::Mat,
    pub flow: core::Mat,
    pub score: core::Mat,
    pub mask: core::Mat,
    pub feat_mask: core::Mat,
    pub dis: core::Ptr<video::DISOpticalFlow>,
    pub kernel_open: core::Mat,
    pub kernel_close: core::Mat,
    pub gray: core::Mat,
    pub ones: core::Mat,
    pub ones_w: core::Mat,
    pub kernel_valid: core::Mat,
    pub morphology: core::Mat,
    pub labels: core::Mat,
    pub stats: core::Mat,
    pub centroids: core::Mat,
    pub h_prev: [f64; 9],
}

impl Ws {
    pub fn new(params: &Params) -> CvResult<Self> {
        params.validate()?;
        let preset = match params.flow_preset {
            FlowPreset::Ultrafast => video::DISOpticalFlow_PRESET_ULTRAFAST,
            FlowPreset::Fast => video::DISOpticalFlow_PRESET_FAST,
            FlowPreset::Medium => video::DISOpticalFlow_PRESET_MEDIUM,
        };
        let mut dis = video::DISOpticalFlow::create(preset)?;
        dis.set_finest_scale(params.finest_scale)?;
        let kernel_open = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            core::Size::new(params.morphology_open_size, params.morphology_open_size),
            core::Point::new(-1, -1),
        )?;
        let kernel_close = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            core::Size::new(params.morphology_close_size, params.morphology_close_size),
            core::Point::new(-1, -1),
        )?;
        let border_width = dis.get_patch_size()?;
        let kernel_valid = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            core::Size::new(2 * border_width + 1, 2 * border_width + 1),
            core::Point::new(-1, -1),
        )?;
        let ones = core::Mat::new_rows_cols_with_default(
            params.h,
            params.w,
            core::CV_8UC1,
            core::Scalar::all(255.0),
        )?;

        Ok(Self {
            prev: core::Mat::default(),
            curr: core::Mat::default(),
            prev_w: core::Mat::default(),
            valid: core::Mat::default(),
            flow: core::Mat::default(),
            score: core::Mat::new_rows_cols_with_default(
                params.h,
                params.w,
                core::CV_32FC1,
                core::Scalar::all(0.0),
            )?,
            mask: core::Mat::new_rows_cols_with_default(
                params.h,
                params.w,
                core::CV_8UC1,
                core::Scalar::all(0.0),
            )?,
            feat_mask: core::Mat::default(),
            dis,
            kernel_open,
            kernel_close,
            gray: core::Mat::default(),
            ones,
            ones_w: core::Mat::default(),
            kernel_valid,
            morphology: core::Mat::default(),
            labels: core::Mat::default(),
            stats: core::Mat::default(),
            centroids: core::Mat::default(),
            h_prev: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], // Identity matrix
        })
    }
}

pub struct Ego {
    pub h: [f64; 9],
    pub f_aligned: Option<[f64; 9]>,
    pub ok: bool,
}

pub trait FrameProcessor {
    fn process_frame(&mut self, frame: &Frame) -> std::result::Result<ProcessingResult, String>;
}

pub struct OpticalFlowProcessor {
    params: Params,
    ws: Ws,
    initialized: bool,
    tracker: Tracker,
    tau_a: f32,
    projected_scores: Vec<f32>,
    feature_pyramids: FeaturePyramids,
    last_ego_reliable: Option<bool>,
}

impl OpticalFlowProcessor {
    pub fn new(params: Params) -> CvResult<Self> {
        let ws = Ws::new(&params)?;
        let tracker = Tracker::new(&params)?;
        let projected_scores = vec![0.0; ws.score.total()];
        Ok(Self {
            tau_a: params.tau_min,
            params,
            ws,
            initialized: false,
            tracker,
            projected_scores,
            feature_pyramids: FeaturePyramids::default(),
            last_ego_reliable: None,
        })
    }

    pub(crate) fn ego_motion_reliable(&self) -> Option<bool> {
        self.last_ego_reliable
    }

    /// Stage 1: Ego-motion (sparse, ~300 points)
    fn estimate_ego(&mut self) -> CvResult<Ego> {
        // We need prev and curr images to be valid
        if self.ws.prev.empty() || self.ws.curr.empty() {
            return Ok(Ego {
                h: self.ws.h_prev,
                f_aligned: None,
                ok: false,
            });
        }

        let mut pts_prev = core::Vector::<core::Point2f>::new();
        opencv::features::good_features_to_track(
            &self.ws.prev,
            &mut pts_prev,
            self.params.max_corners,
            self.params.quality_level,
            self.params.min_distance,
            &self.ws.feat_mask,
            3,
            false,
            0.04,
        )?;

        if pts_prev.is_empty() {
            return Ok(Ego {
                h: self.ws.h_prev,
                f_aligned: None,
                ok: false,
            });
        }

        let mut pts_curr = core::Vector::<core::Point2f>::new();
        let mut status = core::Vector::<u8>::new();
        let mut err = core::Vector::<f32>::new();
        let term_crit =
            core::TermCriteria::new(core::TermCriteria_COUNT + core::TermCriteria_EPS, 20, 0.03)?;

        // Forward LK
        video::calc_optical_flow_pyr_lk(
            &self.feature_pyramids.previous,
            &self.feature_pyramids.current,
            &pts_prev,
            &mut pts_curr,
            &mut status,
            &mut err,
            core::Size::new(21, 21),
            3,
            term_crit,
            0,
            1e-4,
        )?;

        // Backward LK
        let mut pts_prev_back = core::Vector::<core::Point2f>::new();
        let mut status_back = core::Vector::<u8>::new();
        let mut err_back = core::Vector::<f32>::new();
        video::calc_optical_flow_pyr_lk(
            &self.feature_pyramids.current,
            &self.feature_pyramids.previous,
            &pts_curr,
            &mut pts_prev_back,
            &mut status_back,
            &mut err_back,
            core::Size::new(21, 21),
            3,
            term_crit,
            0,
            1e-4,
        )?;

        let mut good_prev = core::Vector::<core::Point2f>::new();
        let mut good_curr = core::Vector::<core::Point2f>::new();

        for i in 0..pts_prev.len() {
            if status.get(i)? == 1 && status_back.get(i)? == 1 {
                let p0 = pts_prev.get(i)?;
                let p0_b = pts_prev_back.get(i)?;
                let dx = p0.x - p0_b.x;
                let dy = p0.y - p0_b.y;
                if (dx * dx + dy * dy).sqrt() < self.params.fb_max {
                    good_prev.push(p0);
                    good_curr.push(pts_curr.get(i)?);
                }
            }
        }

        if good_prev.len() < self.params.min_inliers {
            return Ok(Ego {
                h: self.ws.h_prev,
                f_aligned: None,
                ok: false,
            });
        }

        let mut inliers = core::Mat::default();
        let h_mat = geometry::find_homography(
            &good_prev,
            &good_curr,
            geometry::USAC_MAGSAC,
            self.params.ransac_h,
            &mut inliers,
            2000,
            0.995,
        )?;

        if h_mat.empty() {
            return Ok(Ego {
                h: self.ws.h_prev,
                f_aligned: None,
                ok: false,
            });
        }

        // Convert H to [f64; 9]
        let mut h_arr = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                h_arr[i * 3 + j] = *h_mat.at_2d::<f64>(i as i32, j as i32)?;
            }
        }

        let inliers_data = inliers.data_typed::<u8>()?;
        let num_inliers = inliers_data.iter().filter(|&&v| v != 0).count();
        if num_inliers < self.params.min_inliers {
            return Ok(Ego {
                h: self.ws.h_prev,
                f_aligned: None,
                ok: false,
            });
        }
        let parallax_fraction = 1.0 - (num_inliers as f64 / good_prev.len() as f64);

        let mut f_aligned = None;

        if parallax_fraction > self.params.parallax_threshold {
            let mut f_inliers = core::Mat::default();
            let Some(f_mat) = find_fundamental_matrix(&good_prev, &good_curr, &mut f_inliers)?
            else {
                return Ok(Ego {
                    h: h_arr,
                    f_aligned: None,
                    ok: true,
                });
            };

            if f_mat.rows() == 3 && f_mat.cols() == 3 {
                let mut h_inv_mat = core::Mat::default();
                if core::invert(&h_mat, &mut h_inv_mat, core::DECOMP_LU)? == 0.0 {
                    return Ok(Ego {
                        h: self.ws.h_prev,
                        f_aligned: None,
                        ok: false,
                    });
                }

                let mut f_arr = [0.0; 9];
                let mut h_inv_arr = [0.0; 9];
                for i in 0..3 {
                    for j in 0..3 {
                        f_arr[i * 3 + j] = *f_mat.at_2d::<f64>(i as i32, j as i32)?;
                        h_inv_arr[i * 3 + j] = *h_inv_mat.at_2d::<f64>(i as i32, j as i32)?;
                    }
                }

                f_aligned = Some(mul3(&f_arr, &h_inv_arr));
            }
        }

        Ok(Ego {
            h: h_arr,
            f_aligned,
            ok: true,
        })
    }

    fn stage2_dense_flow(&mut self, ego: &Ego) -> CvResult<()> {
        let size = core::Size::new(self.params.w, self.params.h);

        let mut h_mat =
            core::Mat::new_rows_cols_with_default(3, 3, core::CV_64FC1, core::Scalar::all(0.0))?;
        for i in 0..3 {
            for j in 0..3 {
                *h_mat.at_2d_mut::<f64>(i, j)? = ego.h[(i * 3 + j) as usize];
            }
        }

        imgproc::warp_perspective_def(&self.ws.prev, &mut self.ws.prev_w, &h_mat, size)?;

        imgproc::warp_perspective_def(&self.ws.ones, &mut self.ws.ones_w, &h_mat, size)?;
        imgproc::erode(
            &self.ws.ones_w,
            &mut self.ws.valid,
            &self.ws.kernel_valid,
            core::Point::new(-1, -1),
            1,
            core::BORDER_CONSTANT,
            core::Scalar::default(),
        )?;

        self.ws
            .dis
            .calc(&self.ws.prev_w, &self.ws.curr, &mut self.ws.flow)?;

        Ok(())
    }

    fn stage3_residual_score(&mut self, ego: &Ego) -> CvResult<()> {
        let flow_data = self.ws.flow.data_typed::<core::Vec2f>()?;
        let valid_data = self.ws.valid.data_typed::<u8>()?;
        let pixel_count = usize::try_from(self.params.w)
            .ok()
            .and_then(|width| {
                usize::try_from(self.params.h)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| opencv::Error::new(core::StsOutOfRange, "Invalid processing size"))?;
        if flow_data.len() != pixel_count || valid_data.len() != pixel_count {
            return Err(opencv::Error::new(
                core::StsUnmatchedSizes,
                "Optical-flow buffers do not match the processing size",
            ));
        }

        // 1. Calculate Adaptive noise floor
        let mut mags = Vec::with_capacity(flow_data.len() / 7);
        for i in (0..flow_data.len()).step_by(7) {
            if valid_data[i] != 0 {
                let v = flow_data[i];
                let m = (v[0] * v[0] + v[1] * v[1]).sqrt();
                mags.push(m);
            }
        }

        if ego.ok {
            self.tau_a = adaptive_noise_floor(mags, self.params.tau_min, self.params.k_sigma);
        }
        let tau_a = self.tau_a;
        let gate = self.tracker.gate();

        // 2. Compute Score
        let score_data = self.ws.score.data_typed_mut::<f32>()?;

        residual_scores(
            flow_data,
            valid_data,
            score_data,
            self.params.w as usize,
            (tau_a, self.params.tau_epi),
            gate,
            ego.f_aligned.as_ref(),
        );
        project_scores(
            flow_data,
            valid_data,
            self.ws.score.data_typed::<f32>()?,
            &mut self.projected_scores,
            self.params.w as usize,
        );

        Ok(())
    }

    fn stage4_mask_and_bbox(&mut self) -> CvResult<Option<crate::types::DetectedObject>> {
        // 1. Threshold
        let gate = self.tracker.gate();
        threshold_mask(
            &self.projected_scores,
            self.ws.ones.data_typed::<u8>()?,
            self.ws.mask.data_typed_mut::<u8>()?,
            self.params.w as usize,
            gate,
            self.params.search_score_threshold,
            self.params.tracking_score_threshold,
        );

        // 2. Morphology: OPEN then CLOSE
        imgproc::morphology_ex_def(
            &self.ws.mask,
            &mut self.ws.morphology,
            imgproc::MORPH_OPEN,
            &self.ws.kernel_open,
        )?;
        imgproc::morphology_ex_def(
            &self.ws.morphology,
            &mut self.ws.mask,
            imgproc::MORPH_CLOSE,
            &self.ws.kernel_close,
        )?;

        // 3. Connected Components
        let n_labels = imgproc::connected_components_with_stats_def(
            &self.ws.mask,
            &mut self.ws.labels,
            &mut self.ws.stats,
            &mut self.ws.centroids,
        )?;
        let labels = &self.ws.labels;
        let stats = &self.ws.stats;

        let mut best_label = 0;
        let mut max_score = -1.0;
        let mut best_bbox = None;
        let scores = component_scores(
            labels.data_typed()?,
            &self.projected_scores,
            n_labels as usize,
        );

        for i in 1..n_labels {
            let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
            if area < self.params.min_component_area {
                continue;
            }
            let bounds = core::Rect::new(
                *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?,
                *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?,
                *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?,
                *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?,
            );
            if gate.is_some_and(|gate| (gate & bounds).empty()) {
                continue;
            }

            let sum_score = scores[i as usize];
            if sum_score > max_score {
                max_score = sum_score;
                best_label = i;
                let x = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)? as f32;
                let y = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)? as f32;
                let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)? as f32;
                let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)? as f32;
                best_bbox = Some(crate::types::BoundingBox {
                    x,
                    y,
                    width: w,
                    height: h,
                });
            }
        }

        if best_label > 0 {
            for (pixel, &label) in self
                .ws
                .mask
                .data_bytes_mut()?
                .iter_mut()
                .zip(labels.data_typed::<i32>()?)
            {
                *pixel = u8::from(label == best_label);
            }
        } else {
            self.ws
                .mask
                .set_to(&core::Scalar::all(0.0), &core::no_array())?;
        }

        let best_obj = best_bbox.map(|bbox| crate::types::DetectedObject {
            class_id: 1,
            class_name: "UAV".to_string(),
            confidence: max_score as f32,
            bbox,
        });

        Ok(best_obj)
    }
}

impl FrameProcessor for OpticalFlowProcessor {
    fn process_frame(&mut self, frame: &Frame) -> std::result::Result<ProcessingResult, String> {
        let (_, frame_height, _) = frame.validate_bgr()?;
        let size = core::Size::new(self.params.w, self.params.h);
        let pixel_count = (self.params.w as usize)
            .checked_mul(self.params.h as usize)
            .ok_or_else(|| "Processing resolution is too large".to_string())?;

        let raw_mat = core::Mat::from_slice(&frame.data).map_err(|e| e.to_string())?;
        let frame_mat = raw_mat
            .reshape(3, frame_height)
            .map_err(|e| e.to_string())?;

        if frame_mat.channels() == 3 {
            imgproc::cvt_color_def(&frame_mat, &mut self.ws.gray, imgproc::COLOR_BGR2GRAY)
                .map_err(|e| e.to_string())?;
        } else {
            frame_mat
                .copy_to(&mut self.ws.gray)
                .map_err(|e| e.to_string())?;
        }

        if frame.width == self.params.w as u32 && frame.height == self.params.h as u32 {
            std::mem::swap(&mut self.ws.gray, &mut self.ws.curr);
        } else {
            imgproc::resize(
                &self.ws.gray,
                &mut self.ws.curr,
                size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .map_err(|e| e.to_string())?;
        }

        if !self.initialized {
            self.ws.prev = self.ws.curr.clone();
            self.ws.feat_mask = core::Mat::new_rows_cols_with_default(
                self.params.h,
                self.params.w,
                core::CV_8UC1,
                core::Scalar::all(255.0),
            )
            .map_err(|e| e.to_string())?;
            self.initialized = true;
            return Ok(ProcessingResult {
                objects: vec![],
                mask: crate::types::SemanticMask {
                    data: vec![0; pixel_count],
                    width: self.params.w as u32,
                    height: self.params.h as u32,
                },
            });
        }

        self.feature_pyramids
            .prepare(&self.ws.prev, &self.ws.curr)
            .map_err(|e| e.to_string())?;
        self.ws
            .feat_mask
            .set_to(&core::Scalar::all(255.0), &core::no_array())
            .map_err(|e| e.to_string())?;
        if let Some(target) = self.tracker.feature_exclusion() {
            imgproc::rectangle(
                &mut self.ws.feat_mask,
                target,
                core::Scalar::all(0.0),
                -1,
                imgproc::LINE_8,
                0,
            )
            .map_err(|e| e.to_string())?;
        }
        let ego = self.estimate_ego().map_err(|e| e.to_string())?;
        self.last_ego_reliable = Some(ego.ok);
        if ego.ok {
            self.ws.h_prev = ego.h;
        }

        let ego_to_use = if ego.ok {
            ego
        } else {
            Ego {
                h: self.ws.h_prev,
                f_aligned: None,
                ok: false,
            }
        };

        self.tracker
            .predict(&ego_to_use.h, size)
            .map_err(|e| e.to_string())?;
        self.stage2_dense_flow(&ego_to_use)
            .map_err(|e| e.to_string())?;
        self.stage3_residual_score(&ego_to_use)
            .map_err(|e| e.to_string())?;
        let obj = self.stage4_mask_and_bbox().map_err(|e| e.to_string())?;
        let obj = self
            .tracker
            .update(&self.ws.curr, &mut self.ws.mask, obj, ego_to_use.ok)
            .map_err(|e| e.to_string())?;

        std::mem::swap(&mut self.ws.prev, &mut self.ws.curr);

        let mut objects = vec![];
        if let Some(o) = obj {
            objects.push(o);
        }

        let out_mask_bytes = self.ws.mask.data_bytes().map_err(|e| e.to_string())?;
        if out_mask_bytes.len() != pixel_count {
            return Err(format!(
                "Output mask buffer length mismatch: expected {pixel_count} bytes, got {}",
                out_mask_bytes.len()
            ));
        }
        let mask_data = out_mask_bytes.to_vec();

        Ok(ProcessingResult {
            objects,
            mask: crate::types::SemanticMask {
                data: mask_data,
                width: self.params.w as u32,
                height: self.params.h as u32,
            },
        })
    }
}

fn component_scores(labels: &[i32], scores: &[f32], count: usize) -> Vec<f64> {
    debug_assert_eq!(labels.len(), scores.len());
    let mut sums = vec![0.0; count];
    if count == 1 {
        return sums;
    }
    for (&label, &score) in labels.iter().zip(scores) {
        if label != 0 {
            sums[label as usize] += f64::from(score);
        }
    }
    sums
}

fn find_fundamental_matrix(
    previous: &core::Vector<core::Point2f>,
    current: &core::Vector<core::Point2f>,
    inliers: &mut core::Mat,
) -> CvResult<Option<core::Mat>> {
    match geometry::find_fundamental_mat(
        previous,
        current,
        geometry::USAC_MAGSAC,
        1.0,
        0.999,
        1000,
        inliers,
    ) {
        Ok(matrix) if matrix.empty() => Ok(None),
        Ok(matrix) => Ok(Some(matrix)),
        Err(error)
            if error.code == core::StsAssert
                && error.message.contains("!model.empty()")
                && error.message.contains("setModelParameters") =>
        {
            // USAC can assert instead of returning an empty model on degenerate tracks.
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn residual_scores(
    flow: &[core::Vec2f],
    valid: &[u8],
    scores: &mut [f32],
    width: usize,
    thresholds: (f32, f32),
    gate: Option<core::Rect>,
    f_aligned: Option<&[f64; 9]>,
) {
    debug_assert_eq!(flow.len(), valid.len());
    debug_assert_eq!(flow.len(), scores.len());
    let (tau_a, tau_epi) = thresholds;

    scores
        .par_chunks_mut(width)
        .zip(flow.par_chunks(width))
        .zip(valid.par_chunks(width))
        .enumerate()
        .for_each(|(y, ((score_row, flow_row), valid_row))| {
            for (x, ((score, v), &is_valid)) in score_row
                .iter_mut()
                .zip(flow_row)
                .zip(valid_row)
                .enumerate()
            {
                if is_valid == 0 {
                    *score = 0.0;
                    continue;
                }

                let a = (v[0] * v[0] + v[1] * v[1]).sqrt() / tau_a;
                let (px, py) = (x as f64, y as f64);
                let (qx, qy) = (px + v[0] as f64, py + v[1] as f64);
                let in_gate = gate.is_some_and(|bounds| {
                    qx >= f64::from(bounds.x)
                        && qx < f64::from(bounds.x + bounds.width)
                        && qy >= f64::from(bounds.y)
                        && qy < f64::from(bounds.y + bounds.height)
                });
                let value = match f_aligned {
                    _ if in_gate => a,
                    None => a,
                    Some(f) => {
                        let l0 = f[0] * px + f[1] * py + f[2];
                        let l1 = f[3] * px + f[4] * py + f[5];
                        let l2 = f[6] * px + f[7] * py + f[8];
                        let m0 = f[0] * qx + f[3] * qy + f[6];
                        let m1 = f[1] * qx + f[4] * qy + f[7];
                        let num = qx * l0 + qy * l1 + l2;
                        let d = (num * num / (l0 * l0 + l1 * l1 + m0 * m0 + m1 * m1 + 1e-12)).sqrt()
                            as f32;
                        a.min(d / tau_epi)
                    }
                };
                *score = finite_score(value);
            }
        });
}

fn project_scores(
    flow: &[core::Vec2f],
    valid: &[u8],
    scores: &[f32],
    projected: &mut [f32],
    width: usize,
) {
    debug_assert_eq!(flow.len(), valid.len());
    debug_assert_eq!(flow.len(), scores.len());
    debug_assert_eq!(flow.len(), projected.len());
    projected.fill(0.0);
    if width == 0 {
        return;
    }
    let height = flow.len() / width;

    for (index, ((vector, &is_valid), &score)) in flow.iter().zip(valid).zip(scores).enumerate() {
        if is_valid == 0 || !score.is_finite() {
            continue;
        }
        let x = index % width;
        let y = index / width;
        // shortcut: nearest pixels can leave expansion holes; use area splatting if morphology cannot fill them.
        let target_x = (x as f32 + vector[0]).round();
        let target_y = (y as f32 + vector[1]).round();
        if !target_x.is_finite()
            || !target_y.is_finite()
            || target_x < 0.0
            || target_x >= width as f32
            || target_y < 0.0
            || target_y >= height as f32
        {
            continue;
        }
        let target = target_y as usize * width + target_x as usize;
        projected[target] = projected[target].max(score);
    }
}

fn threshold_mask(
    scores: &[f32],
    valid: &[u8],
    mask: &mut [u8],
    width: usize,
    gate: Option<core::Rect>,
    search_threshold: f32,
    tracking_threshold: f32,
) {
    debug_assert_eq!(scores.len(), valid.len());
    debug_assert_eq!(scores.len(), mask.len());

    mask.par_chunks_mut(width)
        .zip(scores.par_chunks(width))
        .zip(valid.par_chunks(width))
        .enumerate()
        .for_each(|(y, ((mask_row, score_row), valid_row))| {
            for (x, ((value, &score), &is_valid)) in mask_row
                .iter_mut()
                .zip(score_row)
                .zip(valid_row)
                .enumerate()
            {
                let in_gate = gate.is_some_and(|bounds| {
                    let x = x as i32;
                    let y = y as i32;
                    x >= bounds.x
                        && x < bounds.x + bounds.width
                        && y >= bounds.y
                        && y < bounds.y + bounds.height
                });
                let threshold = if in_gate && is_valid != 0 {
                    tracking_threshold
                } else {
                    search_threshold
                };
                *value = if score > threshold { 255 } else { 0 };
            }
        });
}

fn adaptive_noise_floor(mut magnitudes: Vec<f32>, tau_min: f32, k_sigma: f32) -> f32 {
    magnitudes.retain(|magnitude| magnitude.is_finite());
    if magnitudes.is_empty() {
        return tau_min;
    }

    let mid = magnitudes.len() / 2;
    let (_, median, _) = magnitudes.select_nth_unstable_by(mid, f32::total_cmp);
    let median = *median;
    for magnitude in &mut magnitudes {
        *magnitude = (*magnitude - median).abs();
    }
    let (_, mad, _) = magnitudes.select_nth_unstable_by(mid, f32::total_cmp);
    let sigma = 1.4826 * *mad;

    tau_min.max(median + k_sigma * sigma)
}

fn finite_score(score: f32) -> f32 {
    if score.is_finite() {
        score
    } else {
        0.0
    }
}

fn mul3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    fn flow(x: f32, y: f32) -> core::Vec2f {
        core::Vec2f::from([x, y])
    }

    fn serial_residual_scores(
        flow: &[core::Vec2f],
        valid: &[u8],
        scores: &mut [f32],
        width: usize,
        thresholds: (f32, f32),
        gate: Option<core::Rect>,
        f_aligned: Option<&[f64; 9]>,
    ) {
        let (tau_a, tau_epi) = thresholds;
        for y in 0..scores.len() / width {
            for x in 0..width {
                let i = y * width + x;
                if valid[i] == 0 {
                    scores[i] = 0.0;
                    continue;
                }

                let v = flow[i];
                let a = (v[0] * v[0] + v[1] * v[1]).sqrt() / tau_a;
                let (px, py) = (x as f64, y as f64);
                let (qx, qy) = (px + v[0] as f64, py + v[1] as f64);
                let in_gate = gate.is_some_and(|bounds| {
                    qx >= f64::from(bounds.x)
                        && qx < f64::from(bounds.x + bounds.width)
                        && qy >= f64::from(bounds.y)
                        && qy < f64::from(bounds.y + bounds.height)
                });
                let value = match f_aligned {
                    _ if in_gate => a,
                    None => a,
                    Some(f) => {
                        let l0 = f[0] * px + f[1] * py + f[2];
                        let l1 = f[3] * px + f[4] * py + f[5];
                        let l2 = f[6] * px + f[7] * py + f[8];
                        let m0 = f[0] * qx + f[3] * qy + f[6];
                        let m1 = f[1] * qx + f[4] * qy + f[7];
                        let num = qx * l0 + qy * l1 + l2;
                        let d = (num * num / (l0 * l0 + l1 * l1 + m0 * m0 + m1 * m1 + 1e-12)).sqrt()
                            as f32;
                        a.min(d / tau_epi)
                    }
                };
                scores[i] = finite_score(value);
            }
        }
    }

    fn median_ns_per_run(mut run: impl FnMut(), runs_per_sample: usize) -> f64 {
        for _ in 0..2 {
            run();
        }
        let mut samples: Vec<_> = (0..11)
            .map(|_| {
                let started = Instant::now();
                run();
                started.elapsed().as_nanos()
            })
            .collect();
        samples.sort_unstable();
        samples[samples.len() / 2] as f64 / runs_per_sample as f64
    }

    fn benchmark_residual_scores(width: usize, height: usize, f_aligned: Option<&[f64; 9]>) {
        let len = width * height;
        let flow: Vec<_> = (0..len)
            .map(|i| {
                flow(
                    ((i % 17) as f32 - 8.0) * 0.12,
                    ((i % 13) as f32 - 6.0) * 0.15,
                )
            })
            .collect();
        let valid: Vec<_> = (0..len).map(|i| u8::from(i % 31 != 0)).collect();
        let gate = Some(core::Rect::new(
            width as i32 / 3,
            height as i32 / 3,
            width as i32 / 4,
            height as i32 / 4,
        ));
        let thresholds = (0.8, 1.0);
        let runs_per_sample = if len > 500_000 { 2 } else { 5 };
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut serial_output = vec![0.0; len];
        let mut one_thread_output = vec![0.0; len];
        let mut four_thread_output = vec![0.0; len];
        let mut default_output = vec![0.0; len];

        let serial_ns = median_ns_per_run(
            || {
                for _ in 0..runs_per_sample {
                    serial_residual_scores(
                        &flow,
                        &valid,
                        &mut serial_output,
                        width,
                        thresholds,
                        gate,
                        f_aligned,
                    );
                    black_box(&serial_output);
                }
            },
            runs_per_sample,
        );
        let one_thread_ns = median_ns_per_run(
            || {
                one_thread.install(|| {
                    for _ in 0..runs_per_sample {
                        residual_scores(
                            &flow,
                            &valid,
                            &mut one_thread_output,
                            width,
                            thresholds,
                            gate,
                            f_aligned,
                        );
                        black_box(&one_thread_output);
                    }
                });
            },
            runs_per_sample,
        );
        let four_thread_ns = median_ns_per_run(
            || {
                four_threads.install(|| {
                    for _ in 0..runs_per_sample {
                        residual_scores(
                            &flow,
                            &valid,
                            &mut four_thread_output,
                            width,
                            thresholds,
                            gate,
                            f_aligned,
                        );
                        black_box(&four_thread_output);
                    }
                });
            },
            runs_per_sample,
        );
        let default_ns = median_ns_per_run(
            || {
                for _ in 0..runs_per_sample {
                    residual_scores(
                        &flow,
                        &valid,
                        &mut default_output,
                        width,
                        thresholds,
                        gate,
                        f_aligned,
                    );
                    black_box(&default_output);
                }
            },
            runs_per_sample,
        );

        assert!(serial_output
            .iter()
            .zip(&four_thread_output)
            .all(|(left, right)| left.to_bits() == right.to_bits()));
        println!(
            "{width}x{height} epipolar={} median: serial={:.0}ns rayon-1={:.0}ns rayon-4={:.0}ns rayon-default({})={:.0}ns",
            f_aligned.is_some(),
            serial_ns,
            one_thread_ns,
            four_thread_ns,
            rayon::current_num_threads(),
            default_ns,
        );
    }

    #[test]
    #[ignore = "release-only residual-score microbenchmark"]
    fn benchmark_residual_score_parallelism() {
        let fundamental = [0.0, 0.0, 0.01, 0.0, 0.0, -1.0, -0.01, 1.0, 0.0];
        benchmark_residual_scores(640, 360, None);
        benchmark_residual_scores(640, 360, Some(&fundamental));
        benchmark_residual_scores(1280, 720, None);
        benchmark_residual_scores(1280, 720, Some(&fundamental));
    }

    #[test]
    fn residual_scores_apply_fundamental_matrix_outside_gate() {
        let flows = vec![
            flow(3.0, 4.0),
            flow(3.0, 4.0),
            flow(f32::NAN, 1.0),
            flow(1.0, 0.0),
            flow(0.0, 2.0),
            flow(f32::INFINITY, 0.0),
        ];
        let valid = vec![1, 1, 1, 0, 1, 1];
        let mut scores = vec![f32::NAN; flows.len()];
        let fundamental = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];

        residual_scores(
            &flows,
            &valid,
            &mut scores,
            3,
            (1.0, 1.0),
            Some(core::Rect::new(4, 4, 1, 1)),
            Some(&fundamental),
        );

        let epipolar_score = (16.0_f64 / (2.0 + 1e-12)).sqrt() as f32;
        assert_eq!(scores[0].to_bits(), epipolar_score.to_bits());
        assert_eq!(scores[1], 5.0);
        assert_eq!(scores[2], 0.0);
        assert_eq!(scores[3], 0.0);
        let vertical_epipolar_score = (4.0_f64 / (2.0 + 1e-12)).sqrt() as f32;
        assert_eq!(scores[4].to_bits(), vertical_epipolar_score.to_bits());
        assert_eq!(scores[5], 0.0);
    }

    #[test]
    fn residual_scores_use_flow_destination_for_tracking_gate() {
        let flows = vec![
            flow(3.0, 0.0),
            flow(0.0, 0.0),
            flow(-2.0, 0.0),
            flow(0.0, 0.0),
        ];
        let valid = vec![1; flows.len()];
        let mut scores = vec![0.0; flows.len()];
        let fundamental = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];

        residual_scores(
            &flows,
            &valid,
            &mut scores,
            4,
            (1.0, 1.0),
            Some(core::Rect::new(2, 0, 2, 1)),
            Some(&fundamental),
        );

        assert_eq!(scores[0], 3.0);
        assert_eq!(scores[2], 0.0);
    }

    #[test]
    fn projected_scores_land_at_flow_destinations() {
        let mut flows = vec![flow(0.0, 0.0); 12];
        flows[5] = flow(1.0, 1.0);
        flows[6] = flow(0.2, 1.2);
        flows[8] = flow(0.6, -0.6);
        let valid = vec![1; flows.len()];
        let mut scores = vec![0.0; flows.len()];
        scores[0] = 1.0;
        scores[5] = 2.0;
        scores[6] = 3.0;
        scores[8] = 4.0;
        let mut projected = vec![9.0; scores.len()];

        project_scores(&flows, &valid, &scores, &mut projected, 4);

        let mut expected = vec![0.0; scores.len()];
        expected[0] = 1.0;
        expected[5] = 4.0;
        expected[10] = 3.0;
        assert_eq!(projected, expected);
    }

    #[test]
    fn projected_scores_ignore_invalid_non_finite_and_out_of_bounds_flow() {
        let flows = vec![
            flow(f32::NAN, 0.0),
            flow(10.0, 0.0),
            flow(-2.0, 0.0),
            flow(0.0, 0.0),
        ];
        let valid = vec![1, 1, 0, 1];
        let scores = vec![4.0, 3.0, 2.0, 1.0];
        let mut projected = vec![9.0; scores.len()];

        project_scores(&flows, &valid, &scores, &mut projected, 4);

        assert_eq!(projected, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn residual_scores_are_bitwise_stable_across_thread_counts() {
        let width = 64;
        let height = 32;
        let flows: Vec<_> = (0..width * height)
            .map(|i| flow((i % 11) as f32 * 0.25, (i % 7) as f32 * -0.5))
            .collect();
        let valid: Vec<_> = (0..width * height).map(|i| u8::from(i % 13 != 0)).collect();
        let fundamental = [0.0, 0.0, 0.01, 0.0, 0.0, -1.0, -0.01, 1.0, 0.0];
        let gate = Some(core::Rect::new(12, 8, 25, 14));
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut scores = vec![0.0; flows.len()];
            let mut mask = vec![0; flows.len()];
            pool.install(|| {
                residual_scores(
                    &flows,
                    &valid,
                    &mut scores,
                    width,
                    (0.8, 1.25),
                    gate,
                    Some(&fundamental),
                );
                threshold_mask(&scores, &valid, &mut mask, width, gate, 1.0, 0.6);
            });
            (scores, mask)
        };

        let (serial_scores, serial_mask) = run(1);
        let (parallel_scores, parallel_mask) = run(4);
        assert!(serial_scores
            .iter()
            .zip(&parallel_scores)
            .all(|(left, right)| left.to_bits() == right.to_bits()));
        assert_eq!(serial_mask, parallel_mask);
    }

    #[test]
    fn mask_threshold_uses_lower_threshold_for_valid_gate_pixels() {
        let scores = vec![1.0, 1.1, 0.7, 0.6, 0.9, f32::NAN];
        let valid = vec![1, 1, 1, 1, 0, 1];
        let mut mask = vec![99; scores.len()];

        threshold_mask(
            &scores,
            &valid,
            &mut mask,
            3,
            Some(core::Rect::new(0, 1, 3, 1)),
            1.0,
            0.6,
        );

        assert_eq!(mask, vec![0, 255, 0, 0, 0, 0]);
    }

    #[test]
    fn adaptive_noise_floor_ignores_non_finite_values() {
        let threshold = adaptive_noise_floor(
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0, 1.0, 1.0],
            0.8,
            4.0,
        );

        assert_eq!(threshold, 1.0);
    }

    #[test]
    fn adaptive_noise_floor_uses_minimum_when_no_finite_values_exist() {
        let threshold = adaptive_noise_floor(vec![f32::NAN, f32::INFINITY], 0.8, 4.0);

        assert_eq!(threshold, 0.8);
    }

    #[test]
    fn replaces_non_finite_scores_with_zero() {
        assert_eq!(finite_score(f32::NAN), 0.0);
        assert_eq!(finite_score(f32::INFINITY), 0.0);
        assert_eq!(finite_score(1.5), 1.5);
    }

    #[test]
    fn rejects_invalid_processor_parameters() {
        let params = Params {
            w: 0,
            ..Params::default()
        };

        assert!(OpticalFlowProcessor::new(params).is_err());
    }

    #[test]
    fn initial_frame_has_no_ego_estimate_and_textureless_pair_is_unreliable() {
        let mut processor = OpticalFlowProcessor::new(Params {
            w: 64,
            h: 48,
            ..Params::default()
        })
        .unwrap();
        let frame = Frame {
            width: 64,
            height: 48,
            data: vec![100; 64 * 48 * 3],
        };
        processor.process_frame(&frame).unwrap();
        assert_eq!(processor.ego_motion_reliable(), None);
        processor.process_frame(&frame).unwrap();
        assert_eq!(processor.ego_motion_reliable(), Some(false));
    }

    #[test]
    fn degenerate_fundamental_model_is_treated_as_unavailable() -> CvResult<()> {
        let previous: Vec<_> = (0..40)
            .map(|i| core::Point2f::new((i % 8) as f32 * 10.0, (i / 8) as f32 * 10.0))
            .collect();
        let current = vec![core::Point2f::new(20.0, 20.0); previous.len()];

        let result = find_fundamental_matrix(
            &core::Vector::from_slice(&previous),
            &core::Vector::from_slice(&current),
            &mut core::Mat::default(),
        )?;

        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn motion_estimation_recovers_after_textureless_sky() -> CvResult<()> {
        let mut processor = OpticalFlowProcessor::new(Params {
            w: 128,
            h: 128,
            ..Params::default()
        })?;
        let mut sky = Frame {
            width: 128,
            height: 128,
            data: (0..128 * 128)
                .flat_map(|index| {
                    let gradient = (index / 128 / 8) as u8;
                    [220 + gradient, 170 + gradient, 120 + gradient]
                })
                .collect(),
        };
        for exposure in [0, 3, 5] {
            for pixel in &mut sky.data {
                *pixel += exposure;
            }
            let result = processor.process_frame(&sky).unwrap();
            assert!(result.mask.data.iter().all(|&pixel| pixel == 0));
        }
        assert_eq!(processor.ego_motion_reliable(), Some(false));

        let textured = Frame {
            width: 128,
            height: 128,
            data: (0..128 * 128)
                .flat_map(|index| {
                    let (x, y) = (index % 128, index / 128);
                    let value = ((x * 31 + y * 17 + (x * y) % 251) % 256) as u8;
                    [value; 3]
                })
                .collect(),
        };
        processor.process_frame(&textured).unwrap();
        processor.process_frame(&textured).unwrap();
        assert_eq!(processor.ego_motion_reliable(), Some(true));
        Ok(())
    }

    #[test]
    fn valid_fundamental_model_is_preserved() -> CvResult<()> {
        let previous: Vec<_> = (0..40)
            .map(|i| core::Point2f::new((i % 8) as f32 * 10.0, (i / 8) as f32 * 10.0))
            .collect();
        let current: Vec<_> = previous
            .iter()
            .map(|point| core::Point2f::new(point.x + 2.0, point.y + 1.0))
            .collect();

        let result = find_fundamental_matrix(
            &core::Vector::from_slice(&previous),
            &core::Vector::from_slice(&current),
            &mut core::Mat::default(),
        )?;

        assert!(result.is_some_and(|matrix| matrix.rows() == 3 && matrix.cols() == 3));
        Ok(())
    }

    #[test]
    fn unrelated_fundamental_error_is_propagated() {
        let previous =
            core::Vector::from_slice(&[core::Point2f::new(0.0, 0.0), core::Point2f::new(1.0, 1.0)]);
        let current = core::Vector::<core::Point2f>::new();

        let error = find_fundamental_matrix(&previous, &current, &mut core::Mat::default())
            .expect_err("mismatched correspondence counts must remain an error");

        assert!(!error.message.contains("!model.empty()"));
    }

    #[test]
    fn component_scores_match_masked_opencv_sums() -> CvResult<()> {
        let labels = core::Mat::from_slice_2d(&[
            [0, 1, 1, 0, 2, 2],
            [0, 1, 0, 0, 2, 0],
            [3, 0, 0, 4, 0, 0],
        ])?;
        let scores = core::Mat::from_slice_2d(&[
            [9.0f32, 1.25, 2.5, 0.0, 4.0, 1.5],
            [0.0, 0.0, 7.0, 0.0, 2.0, 0.0],
            [0.25, 0.0, 0.0, 16.0, 0.0, 0.0],
        ])?;
        let sums = component_scores(labels.data_typed()?, scores.data_typed()?, 5);
        assert_eq!(sums[0], 0.0);
        for (label, &sum) in sums.iter().enumerate().skip(1) {
            let mut mask = core::Mat::default();
            core::compare(
                &labels,
                &core::Scalar::all(label as f64),
                &mut mask,
                core::CMP_EQ,
            )?;
            let reference =
                core::mean(&scores, &mask)?[0] * f64::from(core::count_non_zero(&mask)?);
            assert_eq!(sum, reference);
        }
        Ok(())
    }

    #[test]
    fn component_selection_preserves_ties_and_clears_reused_masks() -> CvResult<()> {
        let mut processor = OpticalFlowProcessor::new(Params {
            w: 64,
            h: 48,
            morphology_open_size: 1,
            morphology_close_size: 1,
            ..Params::default()
        })?;
        processor.ws.valid = processor.ws.ones.clone();
        let left = core::Rect::new(10, 10, 4, 4);
        let right = core::Rect::new(35, 18, 4, 4);
        for bounds in [left, right] {
            for y in bounds.y..bounds.y + bounds.height {
                let start = y as usize * 64 + bounds.x as usize;
                processor.projected_scores[start..start + bounds.width as usize].fill(1.5);
            }
        }
        let tied = processor.stage4_mask_and_bbox()?.unwrap();
        assert_eq!((tied.bbox.x, tied.bbox.y), (10.0, 10.0));
        assert_eq!(tied.confidence, 24.0);

        for y in right.y..right.y + right.height {
            let start = y as usize * 64 + right.x as usize;
            processor.projected_scores[start..start + right.width as usize].fill(2.0);
        }
        let selected = processor.stage4_mask_and_bbox()?.unwrap();
        assert_eq!((selected.bbox.x, selected.bbox.y), (35.0, 18.0));
        assert_eq!(selected.confidence, 32.0);
        let mask = processor.ws.mask.data_bytes()?;
        assert_eq!(mask.iter().filter(|&&pixel| pixel == 1).count(), 16);
        assert!(mask.iter().all(|&pixel| pixel <= 1));

        processor.projected_scores.fill(0.0);
        assert!(processor.stage4_mask_and_bbox()?.is_none());
        assert!(processor
            .ws
            .mask
            .data_bytes()?
            .iter()
            .all(|&pixel| pixel == 0));
        Ok(())
    }

    #[test]
    fn algorithm_configuration_populates_processor_parameters() {
        let mut config = crate::config::AlgorithmConfig::default();
        config.input_resolution = Some(crate::config::Resolution {
            width: 320,
            height: 180,
        });
        config.min_inliers = 21;
        config.flow_preset = crate::config::FlowPreset::Ultrafast;
        config.morphology_open_size = 7;
        config.search_score_threshold = 1.4;

        let params = Params::from(&config);

        assert_eq!((params.w, params.h), (320, 180));
        assert_eq!(params.min_inliers, 21);
        assert_eq!(params.flow_preset, crate::config::FlowPreset::Ultrafast);
        assert_eq!(params.morphology_open_size, 7);
        assert_eq!(params.search_score_threshold, 1.4);
    }

    #[test]
    fn workspace_uses_configured_morphology_sizes_and_finest_scale() -> CvResult<()> {
        let params = Params {
            morphology_open_size: 7,
            morphology_close_size: 9,
            finest_scale: 2,
            ..Params::default()
        };

        let ws = Ws::new(&params)?;

        assert_eq!(ws.kernel_open.size()?, core::Size::new(7, 7));
        assert_eq!(ws.kernel_close.size()?, core::Size::new(9, 9));
        assert_eq!(ws.dis.get_finest_scale()?, 2);
        Ok(())
    }

    #[test]
    fn mask_thresholds_are_configurable() {
        let scores = vec![0.7, 1.1];
        let valid = vec![1, 1];
        let mut mask = vec![0; 2];

        threshold_mask(
            &scores,
            &valid,
            &mut mask,
            2,
            Some(core::Rect::new(0, 0, 1, 1)),
            1.2,
            0.65,
        );

        assert_eq!(mask, vec![255, 0]);
    }

    #[test]
    fn output_mask_uses_the_processing_resolution() {
        let params = Params {
            w: 12,
            h: 8,
            ..Params::default()
        };
        let mut processor = OpticalFlowProcessor::new(params).unwrap();
        let frame = Frame {
            data: vec![0; 16 * 12 * 3],
            width: 16,
            height: 12,
        };

        let result = processor.process_frame(&frame).unwrap();

        assert_eq!((result.mask.width, result.mask.height), (12, 8));
        assert_eq!(result.mask.data.len(), 12 * 8);
    }
}
