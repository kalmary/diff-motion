use crate::types::{Frame, ProcessingResult};
use opencv::{core, imgproc, video, calib3d, prelude::*, Result as CvResult, features, geometry};

pub struct Params {
    pub w: i32, 
    pub h: i32,
    pub fb_max: f32,
    pub ransac_h: f64,
    pub tau_min: f32,
    pub k_sigma: f32,
    pub tau_epi: f32,
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
        let size = core::Size::new(params.w, params.h);
        
        let dis = video::DISOpticalFlow::create(video::DISOpticalFlow_PRESET_FAST)?;
        let kernel_open = imgproc::get_structuring_element(imgproc::MORPH_RECT, core::Size::new(3, 3), core::Point::new(-1, -1))?;
        let kernel_close = imgproc::get_structuring_element(imgproc::MORPH_RECT, core::Size::new(5, 5), core::Point::new(-1, -1))?;

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
        Ok(Self { params, ws, initialized: false })
    }

    /// Stage 1: Ego-motion (sparse, ~300 points)
    fn estimate_ego(&mut self) -> CvResult<Ego> {
        // We need prev and curr images to be valid
        if self.ws.prev.empty() || self.ws.curr.empty() {
            return Ok(Ego { h: self.ws.h_prev, f_aligned: None, ok: false });
        }

        let mut pts_prev = core::Vector::<core::Point2f>::new();
        imgproc::good_features_to_track(&self.ws.prev, &mut pts_prev, 400, 0.01, 8.0, &self.ws.feat_mask, 3, false, 0.04)?;

        if pts_prev.is_empty() {
            return Ok(Ego { h: self.ws.h_prev, f_aligned: None, ok: false });
        }

        let mut pts_curr = core::Vector::<core::Point2f>::new();
        let mut status = core::Vector::<u8>::new();
        let mut err = core::Vector::<f32>::new();
        let term_crit = core::TermCriteria::new(core::TermCriteria_COUNT + core::TermCriteria_EPS, 20, 0.03)?;

        // Forward LK
        video::calc_optical_flow_pyr_lk(
            &self.ws.prev, &self.ws.curr, &pts_prev, &mut pts_curr, &mut status, &mut err,
            core::Size::new(21, 21), 3, term_crit, 0, 1e-4
        )?;

        // Backward LK
        let mut pts_prev_back = core::Vector::<core::Point2f>::new();
        let mut status_back = core::Vector::<u8>::new();
        let mut err_back = core::Vector::<f32>::new();
        video::calc_optical_flow_pyr_lk(
            &self.ws.curr, &self.ws.prev, &pts_curr, &mut pts_prev_back, &mut status_back, &mut err_back,
            core::Size::new(21, 21), 3, term_crit, 0, 1e-4
        )?;

        let mut good_prev = core::Vector::<core::Point2f>::new();
        let mut good_curr = core::Vector::<core::Point2f>::new();

        for i in 0..pts_prev.len() {
            if status.get(i)? == 1 && status_back.get(i)? == 1 {
                let p0 = pts_prev.get(i)?;
                let p0_b = pts_prev_back.get(i)?;
                let dx = p0.x - p0_b.x;
                let dy = p0.y - p0_b.y;
                if (dx*dx + dy*dy).sqrt() < self.params.fb_max {
                    good_prev.push(p0);
                    good_curr.push(pts_curr.get(i)?);
                }
            }
        }

        if good_prev.len() < 30 {
            return Ok(Ego { h: self.ws.h_prev, f_aligned: None, ok: false });
        }

        let mut inliers = core::Mat::default();
        let h_mat = calib3d::find_homography(&good_prev, &good_curr, calib3d::USAC_MAGSAC, self.params.ransac_h, &mut inliers, 2000, 0.995)?;
        
        if h_mat.empty() {
            return Ok(Ego { h: self.ws.h_prev, f_aligned: None, ok: false });
        }

        // Convert H to [f64; 9]
        let mut h_arr = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                h_arr[i*3 + j] = *h_mat.at_2d::<f64>(i as i32, j as i32)?;
            }
        }

        // Keep as OK for now, fundamental matrix to be added next step.
        Ok(Ego { h: h_arr, f_aligned: None, ok: true })
    }
}
