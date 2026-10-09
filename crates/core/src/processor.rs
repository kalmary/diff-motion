use crate::types::{Frame, ProcessingResult};
use opencv::{core, features, geometry, imgproc, prelude::*, video, Result as CvResult};

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
}

impl Default for Params {
    fn default() -> Self {
        Self {
            w: 640,
            h: 360,
            fb_max: 1.0,
            ransac_h: 1.5,
            tau_min: 0.8,
            k_sigma: 4.0,
            tau_epi: 1.0,
            max_corners: 400,
            quality_level: 0.01,
            min_distance: 8.0,
        }
    }
}

impl Params {
    fn validate(&self) -> CvResult<()> {
        let valid = self.w > 0
            && self.h > 0
            && self.fb_max.is_finite()
            && self.fb_max > 0.0
            && self.ransac_h.is_finite()
            && self.ransac_h > 0.0
            && self.tau_min.is_finite()
            && self.tau_min > 0.0
            && self.k_sigma.is_finite()
            && self.k_sigma >= 0.0
            && self.tau_epi.is_finite()
            && self.tau_epi > 0.0
            && self.max_corners > 0
            && self.quality_level.is_finite()
            && (0.0..=1.0).contains(&self.quality_level)
            && self.quality_level > 0.0
            && self.min_distance.is_finite()
            && self.min_distance >= 0.0;

        if valid {
            Ok(())
        } else {
            Err(opencv::Error::new(
                core::StsBadArg,
                "Invalid optical-flow processor parameters",
            ))
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
    pub h_prev: [f64; 9],
}

impl Ws {
    pub fn new(params: &Params) -> CvResult<Self> {
        params.validate()?;
        let _size = core::Size::new(params.w, params.h);

        let dis = video::DISOpticalFlow::create(video::DISOpticalFlow_PRESET_FAST)?;
        let kernel_open = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            core::Size::new(3, 3),
            core::Point::new(-1, -1),
        )?;
        let kernel_close = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            core::Size::new(5, 5),
            core::Point::new(-1, -1),
        )?;

        Ok(Self {
            prev: core::Mat::default(),
            curr: core::Mat::default(),
            prev_w: core::Mat::default(),
            valid: core::Mat::default(),
            flow: core::Mat::default(),
            score: core::Mat::default(),
            mask: core::Mat::default(),
            feat_mask: core::Mat::default(),
            dis,
            kernel_open,
            kernel_close,
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
}

impl OpticalFlowProcessor {
    pub fn new(params: Params) -> CvResult<Self> {
        let ws = Ws::new(&params)?;
        Ok(Self {
            params,
            ws,
            initialized: false,
        })
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
        features::good_features_to_track(
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
            &self.ws.prev,
            &self.ws.curr,
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
            &self.ws.curr,
            &self.ws.prev,
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

        if good_prev.len() < 30 {
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
        let parallax_fraction = 1.0 - (num_inliers as f64 / good_prev.len() as f64);

        let mut f_aligned = None;

        if parallax_fraction > 0.08 {
            let mut f_inliers = core::Mat::default();
            let f_mat = geometry::find_fundamental_mat(
                &good_prev,
                &good_curr,
                geometry::USAC_MAGSAC,
                1.0,
                0.999,
                1000,
                &mut f_inliers,
            )?;

            if !f_mat.empty() && f_mat.rows() == 3 && f_mat.cols() == 3 {
                let mut h_inv_mat = core::Mat::default();
                core::invert(&h_mat, &mut h_inv_mat, core::DECOMP_LU)?;

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

        let ones = core::Mat::new_rows_cols_with_default(
            self.params.h,
            self.params.w,
            core::CV_8UC1,
            core::Scalar::all(255.0),
        )?;
        let mut ones_w = core::Mat::default();
        imgproc::warp_perspective_def(&ones, &mut ones_w, &h_mat, size)?;

        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            core::Size::new(7, 7),
            core::Point::new(-1, -1),
        )?;
        imgproc::erode(
            &ones_w,
            &mut self.ws.valid,
            &kernel,
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

        let tau_a = adaptive_noise_floor(mags, self.params.tau_min, self.params.k_sigma);

        // 2. Compute Score
        self.ws.score = core::Mat::new_rows_cols_with_default(
            self.params.h,
            self.params.w,
            core::CV_32FC1,
            core::Scalar::all(0.0),
        )?;
        let score_data = self.ws.score.data_typed_mut::<f32>()?;

        let ww = self.params.w as usize;
        let hh = self.params.h as usize;

        for y in 0..hh {
            for x in 0..ww {
                let i = y * ww + x;
                if valid_data[i] == 0 {
                    score_data[i] = 0.0;
                    continue;
                }
                let v = flow_data[i];
                let a = (v[0] * v[0] + v[1] * v[1]).sqrt() / tau_a;

                let score = match &ego.f_aligned {
                    None => a,
                    Some(f) => {
                        let (px, py) = (x as f64, y as f64);
                        let (qx, qy) = (px + v[0] as f64, py + v[1] as f64);
                        let l0 = f[0] * px + f[1] * py + f[2];
                        let l1 = f[3] * px + f[4] * py + f[5];
                        let l2 = f[6] * px + f[7] * py + f[8];
                        let m0 = f[0] * qx + f[3] * qy + f[6];
                        let m1 = f[1] * qx + f[4] * qy + f[7];
                        let num = qx * l0 + qy * l1 + l2;
                        let d = (num * num / (l0 * l0 + l1 * l1 + m0 * m0 + m1 * m1 + 1e-12)).sqrt()
                            as f32;
                        a.min(d / self.params.tau_epi)
                    }
                };
                score_data[i] = finite_score(score);
            }
        }

        Ok(())
    }

    fn stage4_mask_and_bbox(&mut self) -> CvResult<Option<crate::types::DetectedObject>> {
        // 1. Threshold
        imgproc::threshold(
            &self.ws.score,
            &mut self.ws.mask,
            1.0,
            255.0,
            imgproc::THRESH_BINARY,
        )?;
        let mut mask_8u = core::Mat::default();
        self.ws
            .mask
            .convert_to(&mut mask_8u, core::CV_8UC1, 1.0, 0.0)?;
        self.ws.mask = mask_8u;

        // 2. Morphology: OPEN then CLOSE
        let mut temp = core::Mat::default();
        imgproc::morphology_ex_def(
            &self.ws.mask,
            &mut temp,
            imgproc::MORPH_OPEN,
            &self.ws.kernel_open,
        )?;
        imgproc::morphology_ex_def(
            &temp,
            &mut self.ws.mask,
            imgproc::MORPH_CLOSE,
            &self.ws.kernel_close,
        )?;

        // 3. Connected Components
        let mut labels = core::Mat::default();
        let mut stats = core::Mat::default();
        let mut centroids = core::Mat::default();

        let n_labels = imgproc::connected_components_with_stats_def(
            &self.ws.mask,
            &mut labels,
            &mut stats,
            &mut centroids,
        )?;

        let mut best_label = 0;
        let mut max_score = -1.0;
        let mut best_bbox = None;

        for i in 1..n_labels {
            let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
            if area < 12 {
                // A_min
                continue;
            }

            // To get sum over mask: mean * area
            let mut comp_mask = core::Mat::default();
            core::compare(
                &labels,
                &core::Scalar::all(i as f64),
                &mut comp_mask,
                core::CMP_EQ,
            )?;
            let mean_score = core::mean(&self.ws.score, &comp_mask)?;
            let sum_score = mean_score[0] * area as f64;

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

        let mut mask_8u_final = core::Mat::default();
        if best_label > 0 {
            core::compare(
                &labels,
                &core::Scalar::all(best_label as f64),
                &mut self.ws.mask,
                core::CMP_EQ,
            )?;
            self.ws
                .mask
                .convert_to(&mut mask_8u_final, core::CV_8UC1, 1.0 / 255.0, 0.0)?;
            self.ws.mask = mask_8u_final;
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
        let (frame_width, frame_height, pixel_count) = frame.validate_bgr()?;
        let size = core::Size::new(self.params.w, self.params.h);

        let raw_mat = core::Mat::from_slice(&frame.data).map_err(|e| e.to_string())?;
        let frame_mat = raw_mat
            .reshape(3, frame_height)
            .map_err(|e| e.to_string())?;

        let mut curr_gray = core::Mat::default();
        if frame_mat.channels() == 3 {
            imgproc::cvt_color_def(&frame_mat, &mut curr_gray, imgproc::COLOR_BGR2GRAY)
                .map_err(|e| e.to_string())?;
        } else {
            frame_mat
                .copy_to(&mut curr_gray)
                .map_err(|e| e.to_string())?;
        }

        imgproc::resize(
            &curr_gray,
            &mut self.ws.curr,
            size,
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )
        .map_err(|e| e.to_string())?;

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
                    width: frame.width,
                    height: frame.height,
                },
            });
        }

        let ego = self.estimate_ego().map_err(|e| e.to_string())?;
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

        self.stage2_dense_flow(&ego_to_use)
            .map_err(|e| e.to_string())?;
        self.stage3_residual_score(&ego_to_use)
            .map_err(|e| e.to_string())?;
        let obj = self.stage4_mask_and_bbox().map_err(|e| e.to_string())?;

        self.ws.prev = self.ws.curr.clone();

        // Scale bbox back to original frame size
        let mut objects = vec![];
        if let Some(mut o) = obj {
            let scale_x = frame.width as f32 / self.params.w as f32;
            let scale_y = frame.height as f32 / self.params.h as f32;
            o.bbox.x *= scale_x;
            o.bbox.y *= scale_y;
            o.bbox.width *= scale_x;
            o.bbox.height *= scale_y;
            objects.push(o);
        }

        let mut out_mask = core::Mat::default();
        imgproc::resize(
            &self.ws.mask,
            &mut out_mask,
            core::Size::new(frame_width, frame_height),
            0.0,
            0.0,
            imgproc::INTER_NEAREST,
        )
        .map_err(|e| e.to_string())?;

        let out_mask_bytes = out_mask.data_bytes().map_err(|e| e.to_string())?;
        if out_mask_bytes.len() != pixel_count {
            return Err(format!(
                "Output mask buffer length mismatch: expected {pixel_count} bytes, got {}",
                out_mask_bytes.len()
            ));
        }
        let mut mask_data = vec![0u8; pixel_count];
        mask_data.copy_from_slice(out_mask_bytes);

        Ok(ProcessingResult {
            objects,
            mask: crate::types::SemanticMask {
                data: mask_data,
                width: frame.width,
                height: frame.height,
            },
        })
    }
}

fn adaptive_noise_floor(mut magnitudes: Vec<f32>, tau_min: f32, k_sigma: f32) -> f32 {
    magnitudes.retain(|magnitude| magnitude.is_finite());
    if magnitudes.is_empty() {
        return tau_min;
    }

    let mid = magnitudes.len() / 2;
    let (_, median, _) = magnitudes.select_nth_unstable_by(mid, f32::total_cmp);
    let median = *median;
    let mut absolute_deviations: Vec<f32> = magnitudes
        .iter()
        .map(|magnitude| (magnitude - median).abs())
        .collect();
    let (_, mad, _) = absolute_deviations.select_nth_unstable_by(mid, f32::total_cmp);
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
}
