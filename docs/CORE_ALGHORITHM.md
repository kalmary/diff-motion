# Deterministic Motion-Based UAV Segmentation (Rust + OpenCV, Jetson Nano)

Purely deterministic pipeline: **ego-motion compensation + aligned dense optical flow + epipolar residual test + tracker state machine**. No neural networks, no learned models.

Output per frame: a binary mask (target = 1, background = 0) and a bounding box derived from it.

> Status: design spec. Not compiled or benchmarked. Check `opencv` crate signatures against the pinned version (`find_homography`, `cudaoptflow` wrappers change between releases). All latency figures are **budgets to be measured**, not results.

---

## 0. Plan (each step has a check)

| # | Step | Verify |
|---|------|--------|
| 1 | Ego-motion `H` from sparse LK + RANSAC | Background inlier residual < 1 px on clips with a static target |
| 2 | Warp `prev` by `H`, dense flow on aligned pair | Background residual flow is noise-like, median < 0.3 px |
| 3 | Residual score: magnitude cue + epipolar cue | Static scene with parallax yields no blobs |
| 4 | Threshold, morphology, one connected component, bbox | IoU on synthetic injected targets |
| 5 | Tracker state machine (SEARCH, LOCK, COAST) | Target held through a forced zero-residual segment |

---

## 1. Core idea

Do **not** run dense flow on raw frames. Warp `prev` into `curr` coordinates with the estimated homography `H` first. Dense flow on the aligned pair only has to find small residual motion, which is where Farneback and DIS work best and cost least.

The residual then contains two things:

1. The target.
2. Parallax from background that is off the dominant plane.

A homography alone cannot separate them. The **epipolar constraint** can:

- Static background at *any* depth satisfies `x'ᵀ F x = 0` after ego-motion. Parallax slides points along their epipolar lines, so it cannot violate the equation.
- An independently moving object generally violates it.

This is what kills false positives on parallax edges.

---

## 2. Data layout (allocate once, reuse every frame)

```rust
use opencv::{core::*, imgproc, video, calib3d, prelude::*, Result};

pub struct Params {
    pub w: i32, pub h: i32,   // working res, e.g. 640x360 (try 480x270 on Nano)
    pub fb_max: f32,          // forward-backward LK gate, 1.0 px
    pub ransac_h: f64,        // 1.5 px
    pub tau_min: f32,         // residual floor, 0.8 px
    pub k_sigma: f32,         // 4.0
    pub tau_epi: f32,         // epipolar px, 1.0
}

pub struct Ws {
    prev: Mat, curr: Mat,     // CV_8UC1
    prev_w: Mat,              // CV_8UC1, prev warped into curr coords
    valid: Mat,               // CV_8UC1, warped-ones mask, eroded
    flow: Mat,                // CV_32FC2, residual flow
    score: Mat,               // CV_32FC1, fused score, 1.0 = threshold
    mask: Mat,                // CV_8UC1
    feat_mask: Mat,           // CV_8UC1, 0 inside tracker gate
    dis: core::Ptr<video::DISOpticalFlow>,
    kernel_open: Mat, kernel_close: Mat,
    h_prev: [f64; 9],
}

pub struct Ego { pub h: [f64; 9], pub f_aligned: Option<[f64; 9]>, pub ok: bool }
```

Parameter table:

| Name | Default | Meaning |
|------|---------|---------|
| working res | 640x360 | Do not go much below 480 px wide: tiny targets vanish in flow smoothing |
| `fb_max` | 1.0 px | Forward-backward LK consistency gate |
| `ransac_h` | 1.5 px | Homography RANSAC reprojection threshold |
| `tau_min` | 0.8 px | Lower bound for the residual magnitude threshold |
| `k_sigma` | 4.0 | Multiplier on robust noise sigma |
| `tau_epi` | 1.0 px | Sampson distance normalisation |
| `A_min` | 12 px | Minimum blob area at working res |

---

## 3. Stage 1: Ego-motion (sparse, ~300 points)

1. **Features.** `good_features_to_track(prev, max=400, quality=0.01, minDist=8)` with `feat_mask` zeroing the tracker's predicted target box. Bucket into an 8x6 grid, at most ~8 points per cell, so one textured building cannot dominate the fit.
2. **Forward-backward LK.** `calc_optical_flow_pyr_lk(prev -> curr)` then `(curr -> prev)`. Keep points with `status == 1` and `|p0 - p0_back| < fb_max`. Window 21x21, 3 pyramid levels, criteria `(COUNT|EPS, 20, 0.03)`. This removes aperture-problem and occlusion garbage deterministically.
3. **Homography.** `find_homography(src, dst, inliers, USAC_MAGSAC, 1.5)`. The target is a small outlier minority, so RANSAC ignores it. If fewer than 30 inliers: set `ok = false`, reuse `h_prev`, freeze the threshold update.
4. **Fundamental matrix** (only when parallax is significant): `find_fundamental_mat(src, dst, USAC_MAGSAC, 1.0, 0.999)` on the FB-passed points.
   - Parallax measure = fraction of FB-passed points whose H-residual exceeds 1.5 px.
   - If below ~8%: rotation-dominant camera or flat scene, `F` is degenerate. Skip the epipolar cue, use magnitude only (`f_aligned = None`).
   - Otherwise use both cues.
5. **Move `F` into the aligned frame.** The dense flow lives in aligned coordinates `y = Hx`, so use `F' = F * H^-1`. Per pixel, with `q = y + w(y)`, the epipolar error is `qᵀ F' y`.

```rust
fn mul3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut r = [0.0; 9];
    for i in 0..3 { for j in 0..3 { for k in 0..3 {
        r[i*3 + j] += a[i*3 + k] * b[k*3 + j];
    }}}
    r
}
// f_aligned = mul3(&F, &H_inv)   (invert H via Mat::inv or closed-form 3x3)
```

---

## 4. Stage 2: Aligned dense flow

```rust
imgproc::warp_perspective(&ws.prev, &mut ws.prev_w, &h_mat, size,
    imgproc::INTER_LINEAR, BORDER_CONSTANT, Scalar::default())?;
// valid region: warp an all-255 image once with the same H, erode 7x7 (kills border artefacts)
ws.dis.calc(&ws.prev_w, &ws.curr, &mut ws.flow)?;
```

- **CPU default:** `DISOpticalFlow`, preset `ULTRAFAST` or `FAST`. Much cheaper than Farneback on a Nano and deterministic. Residuals are small after alignment, so DIS's weaker handling of large motion costs nothing.
- **Fallback:** `calc_optical_flow_farneback(.., 0.5, 3, 15, 3, 5, 1.2, 0)`.

---

## 5. Stage 3: Residual score (fused)

**Adaptive noise floor.** Subsample residual magnitude (every 7th pixel inside `valid`), take median `m` and `MAD`, set `sigma = 1.4826 * MAD`, then:

```
tau_A = max(tau_min, m + k_sigma * sigma)
```

Use `select_nth_unstable_by`, not a full sort. This self-tunes to jitter, wind buffeting and blur.

**Per-pixel score.** Plain loop over slices (~0.2M pixels at working res, no SIMD needed):

```rust
let flow = ws.flow.data_typed::<Vec2f>()?;
let sc   = ws.score.data_typed_mut::<f32>()?;
let val  = ws.valid.data_bytes()?;
for y in 0..hh { for x in 0..ww {
    let i = (y * ww + x) as usize;
    if val[i] == 0 { sc[i] = 0.0; continue; }
    let v = flow[i];
    let a = (v[0]*v[0] + v[1]*v[1]).sqrt() / tau_a;            // cue A: magnitude
    sc[i] = match &ego.f_aligned {
        None => a,
        Some(f) => {
            let (px, py) = (x as f64, y as f64);
            let (qx, qy) = (px + v[0] as f64, py + v[1] as f64);
            let l0 = f[0]*px + f[1]*py + f[2];                   // F' p
            let l1 = f[3]*px + f[4]*py + f[5];
            let l2 = f[6]*px + f[7]*py + f[8];
            let m0 = f[0]*qx + f[3]*qy + f[6];                   // F'^T q
            let m1 = f[1]*qx + f[4]*qy + f[7];
            let num = qx*l0 + qy*l1 + l2;
            let d = (num*num / (l0*l0 + l1*l1 + m0*m0 + m1*m1 + 1e-12)).sqrt() as f32; // Sampson
            a.min(d / tau_epi)                                    // soft AND
        }
    };
}}
```

`min(A, B)` is a soft AND: a pixel scores high only if it moves relative to the aligned background **and** violates epipolar geometry.

---

## 6. Stage 4: Mask and bounding box

1. `mask = score > 1.0`, or `> 0.6` inside the tracker gate (hysteresis).
2. `morphology_ex`: OPEN 3x3 (removes speckle and single-pixel jitter), then CLOSE 5x5 (fills holes in the target).
3. `connected_components_with_stats`. Drop components with area < `A_min`, extreme aspect ratio, or low solidity (thin parallax slivers).
4. Select one component:
   - SEARCH: highest `sum(score)` over the component.
   - LOCK: best IoU or distance to the Kalman prediction.
5. Zero all other pixels. This is the binary mask.
6. Bounding box from that component's stats, scaled back to full resolution.

---

## 7. Stage 5: Stagnation and zero-delta edge cases

When the target's apparent motion equals the background's, either:

- **(a)** it moves exactly with the camera: the residual is zero, or
- **(b)** it moves along its own epipolar line: the epipolar cue is zero even though magnitude is positive.

A single-frame motion method cannot see (a). Handle it with state, not a lower threshold.

| State | Entry | Behavior |
|-------|-------|----------|
| SEARCH | no target | Full-frame score, strict thresholds, pick best component |
| LOCK | blob stable in >= 3 of 5 frames | Constant-velocity Kalman (center, size, velocity) in aligned coordinates: transform previous center by `H` each frame (`perspective_transform`), then predict. Gate = inflated predicted bbox. Inside the gate: relaxed threshold, **cue A only** (covers case b). |
| COAST | residual area collapses inside the gate | `match_template` (`TM_CCOEFF_NORMED`) with a patch saved at LOCK, searched inside the gate. Mask = last mask warped by `H` and shifted to the match location. Kalman keeps predicting. |
| COAST -> SEARCH | COAST > ~15 frames, or ZNCC < 0.5 | Drop the track |
| COAST -> LOCK | strong blob overlaps the gate | Reacquire |

Optional for slow relative motion: also compute residual flow against frame `t-k` (`k = 2..4`, ring buffer of aligned frames) and take the max score over baselines. A longer baseline multiplies displacement without changing noise.

---

## 8. Why false positives stay low

| Source | Mitigation |
|--------|-----------|
| Camera jitter, attitude change | Removed by per-frame `H`; adaptive MAD threshold scales with leftover noise |
| Parallax edges | Sampson term is ~0 for any static depth; shape/area filter removes leftover slivers |
| Warp border artefacts | Eroded `valid` mask |
| Bad ego estimate | Inlier-count gate freezes previous `H`; no new tracks start while `ok = false` |
| One-frame flicker | 3-of-5 temporal consistency before LOCK |
| Target pixels polluting ego fit | Features excluded from tracker gate; RANSAC treats the target as outliers |

---

## 9. CUDA path (behind `--features cuda`)

- Build OpenCV with contrib and `WITH_CUDA=ON`, `CUDA_ARCH_BIN=5.3` (Nano). The apt package has no CUDA.
- Port with `GpuMat`: `cudawarping::warp_perspective`, `cudaoptflow::FarnebackOpticalFlow` (or `DensePyrLKOpticalFlow`), `cudaimgproc` corner detection.
- Keep RANSAC fits and the score loop on CPU. Nano has unified memory, so downloading the flow is cheap.
- One trait boundary only where the flow call differs (`fn flow(&mut self, a, b) -> flow`). No backend framework.

---

## 10. Per-frame pseudocode

```
frame -> gray -> resize(working res)
ego   = estimate_ego(prev, curr, gate)            // Stage 1
if !ego.ok { reuse h_prev }
prev_w = warp(prev, H); valid = warp(ones, H) eroded
flow   = DIS(prev_w, curr)                        // Stage 2
tau_a  = median + k*1.4826*MAD(|flow|)
score  = fuse(|flow|/tau_a, sampson/tau_epi)      // Stage 3
mask   = threshold -> open -> close -> 1 component // Stage 4
state  = tracker.update(mask, H)                  // Stage 5
bbox   = bbox(mask) scaled to full res
prev   = curr
```

---

## 11. Verification (numbers to measure)

- **Synthetic injection.** Take real footage, warp with known random `H` plus jitter, paste a moving patch at known positions. Report IoU against ground-truth mask and false blobs per frame.
- **Parallax clip.** Static target, fast forward flight past buildings and trees. False-blob rate should be ~0 with the epipolar cue on, visibly worse with `F` disabled (proves the cue pays for itself).
- **Stagnation clip.** Force zero relative motion for N frames. Report max N held and reacquire rate.
- **Latency.** Per-stage timing with `tegrastats` running. Budgets at working res: ego < 5 ms, flow < 12 ms, score + mask < 3 ms.