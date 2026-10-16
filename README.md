# Diff Motion

**Diff Motion** is a real-time computer-vision application designed to run on an NVIDIA Jetson Nano mounted on an unmanned aerial vehicle (UAV). It detects a target UAV purely based on relative motion discrepancy (optical flow separation) compared to the background. 

This project uses a **strictly deterministic algorithm** based on ego-motion compensation, dense optical flow, and epipolar residual testing. **No neural networks, deep learning models, or AI are used for the segmentation.**

## Features

- **Purely Deterministic:** No AI models. Relies entirely on classical computer vision (Optical Flow, RANSAC, Epipolar geometry).
- **Headless & Display Modes:** Can run entirely headlessly on the Jetson Nano, or display the camera image with a semi-transparent red semantic mask overlay.
- **Cross-Platform & Highly Performant:** Written in Rust, leveraging OpenCV 5.0 C++ bindings for maximum performance.
- **Python Bindings:** Includes Python bindings via PyO3 for integration with Python-based drone control systems.

## Architecture

The application is a Cargo workspace with three crates around the same processing pipeline:

- **`crates/core` (`diff-motion-core`)** contains capture, OpenCV processing, tracking, masks, and bounding boxes.
- **`crates/cli` (`diff-motion-cli`)** runs the pipeline as a standalone command.
- **`crates/python` (`diff-motion-python`)** exposes the pipeline as `import diff_motion` through PyO3 and Maturin.

## Prerequisites

- **Conda** (Miniconda or Anaconda; provides the project's Rust/Cargo toolchain and native libraries)
- **Host C/C++ toolchain** (e.g., Xcode Command Line Tools on macOS, Visual Studio Build Tools and LLVM on Windows, or Clang and `libclang-dev` on Linux)
- **uv** (Python interpreter and project environment manager; use the installation method supported by your operating system)
- **Just** (cross-platform command runner; install it independently or with your system package manager)

### Environment Setup

Native libraries and Rust share the Conda environment. Python remains isolated in the uv environment:

| Layer | Owner | Contents |
| --- | --- | --- |
| Native C/C++ dependencies | Conda | `libopencv` 5, `pkg-config`, and CMake in the `diff-motion` environment. |
| Rust | Conda | Rust 1.99.0, including Cargo, rustc, rustfmt, and Clippy, in the `diff-motion` environment. |
| Python | uv | The managed Python interpreter, `.venv`, Maturin, pytest, and other recorded Python dependencies. |

Conda may install Python as a transitive dependency of a native package. The project never uses
that interpreter or installs its Python packages into Conda. Python builds, tests, and runtime use
the uv-managed `.venv`.

Prepare the Conda Rust/native environment and the uv Python environment:

```bash
just setup
```

`setup` covers these stages; `setup-rust` depends on `setup-conda`:

```bash
just setup-conda
just setup-rust
just setup-python
```

| Stage | Result |
| --- | --- |
| `just setup-conda` | Creates or updates `diff-motion` with Rust 1.99.0 and the native OpenCV/build dependencies, then clears cached OpenCV bindings in debug and release builds. |
| `just setup-rust` | Ensures the Conda environment exists, checks its Cargo, rustc, rustfmt, and Clippy, then fetches locked Cargo dependencies through `conda run`. It does not depend on shell activation or a global Rust installation. |
| `just setup-python` | Runs `uv sync --managed-python --locked --no-install-project --group test`, creating `.venv` with development Maturin and test-only pytest. It does not build the extension. |

`environment.yml` pins the Conda `rust` package to exactly 1.99.0. That package includes Cargo,
so Cargo has no separate toolchain-version requirement. This is a compiler pin rather than a lock
of every native Conda package; `Cargo.lock` and `uv.lock` freeze the recorded Rust and Python
package graphs. The crate manifests continue to declare Rust 1.88 as their minimum supported
Rust version. Platform packages for Rust 1.99.0 are published by the
[conda-forge Rust feedstock](https://github.com/conda-forge/rust-feedstock).

Python setup does not use global or system package installation. Maturin remains the development
CLI and declared build backend; pytest belongs to the test dependency group. Build and test
workflows use uv's editable project sync to invoke the Maturin backend and install the extension
into `.venv`. Build isolation is disabled because the recorded Maturin CLI is already installed
there by `setup-python`.

Setup prepares the tools. Run `just build` after setup to make `diff_motion` importable in `.venv`.

Pyright and basedpyright use the project's `.venv` through the relative configuration in
`pyproject.toml`. The Python extension includes type information from `diff_motion.pyi`.
If an editor still reports unresolved imports after setup and build, select the project `.venv`
as its Python interpreter and restart its Python language server.

After cleaning Cargo artifacts, rust-analyzer may retain paths to deleted procedural-macro
libraries under `target/debug`. Recreate them with `cargo check --workspace --all-targets --all-features`
inside `just activate`, then restart the editor's Rust language server. In Zed, run
`editor: restart language server` from the command palette while a Rust file is open.

For a terminal ready to build, test, and run the whole project, use:

```bash
just activate
```

`just activate` opens a child shell with the native Conda environment and uv `.venv` active.
Cargo and rustc come from that Conda environment without a separate Rust activation step.
Inside that shell, run `just build`, `just test`, or `just run`. Type `exit` to
return to your original terminal.

The individual commands provide narrower entry points:

| Command | What you enter | What it selects |
| --- | --- | --- |
| `just activate` | A terminal shell | Native dependencies and pinned Rust/Cargo from Conda, plus the uv `.venv`. |
| `just activate-conda` | A terminal shell | Native dependencies and pinned Rust/Cargo from Conda. |
| `just activate-python` | A terminal shell | The uv `.venv` only. |

For manual activation, activate the native Conda environment and uv virtual environment in the
current shell. On Unix shells:

```bash
conda activate diff-motion
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
conda activate diff-motion
.venv\Scripts\Activate.ps1
```

The combined `just activate` command selects the native libraries and Rust/Cargo toolchain from
Conda and Python from uv. `just activate-python` intentionally selects only the uv environment.

Run setup explicitly after changing `environment.yml` or Python dependency groups.
Native setup clears cached Rust OpenCV bindings so the next build discovers the current
Conda libraries instead of reusing bindings linked to a previous environment or system installation.
Routine build and test workflows do not provision Conda dependencies;
their locked uv sync rebuilds and reinstalls the Python extension in `.venv`.

If Maturin reports a missing `@rpath/libabsl_*.dylib`, cached OpenCV bindings may
still reference another installation, such as Homebrew. From `just activate`, clear
those bindings and rebuild with the project toolchain:

```bash
cargo clean --package opencv
cargo clean --release --package opencv
just build
```

Keep builds and editor build commands inside the project Conda environment so cached
native bindings continue to use its OpenCV. Reinstalling Python packages alone does
not clear Cargo's native dependency cache.

## Building & Testing

The public Just recipes group operations across Rust, Python, and native dependencies:

| Command | Description |
| --- | --- |
| `just setup` | Prepare the Conda Rust/native environment and uv Python environment. |
| `just setup-conda` | Create or update the environment with native dependencies and pinned Rust/Cargo. |
| `just setup-rust` | Check Rust tools inside Conda and fetch locked Cargo dependencies. |
| `just setup-python` | Create or update the uv `.venv` with development and test tools. |
| `just activate` | Open a project shell with Conda native/Rust and uv Python active. |
| `just activate-conda` | Open a shell with Conda native dependencies and Rust/Cargo active. |
| `just activate-python` | Open a shell with the uv `.venv` active. |
| `just clean-python` | Run `uv cache clean` and remove the project's `.venv`. |
| `just clean-conda` | Remove the project's Rust/native Conda environment. |
| `just clean-rust` | Remove workspace Cargo build artifacts using Cargo from Conda. |
| `just clean` | Clean Cargo artifacts, uv's shared cache, `.venv`, and the project Conda environment. |
| `just clean-all` | Alias for `just clean`. |
| `just build` | Build the Rust application and Python extension. |
| `just build-headless` | Build both entry points without OpenCV display support. |
| `just test` | Build the Python extension and run the Rust and Python test suites. |
| `just test-headless` | Run both test suites without OpenCV display support. |
| `just verify` | Verify formatting, linting, builds, and tests with and without display support. |
| `just run <arguments>` | Run the CLI from the prepared project environment. |

Build, test, and verify recipes require both the native Conda environment and uv `.venv` to
be active. The Rust-only `run` recipe requires the native Conda environment, without Python
activation. These recipes verify that rustc and Cargo resolve inside `diff-motion`, rejecting a
global fallback. Tests call `python -m pytest`, and Python extension builds use uv's editable sync
with the Maturin backend. This keeps the Conda Rust/native tools available while selecting the
uv-managed Python explicitly, without installing Python packages into Conda or the system interpreter.

Cleanup commands work inside or outside the activated project shell. `just clean-rust` invokes
Cargo through `conda run`, so it runs before Conda removal. `just clean-conda` runs the Conda
manager from `base`, allowing removal even when the parent shell has `diff-motion` active.
If Conda is already absent, Rust cleanup removes the default workspace `target` directory and
Conda cleanup does nothing. Cleanup can be repeated. Python cleanup runs `uv cache clean`, which
clears the shared uv package/build cache used by other projects too, then removes `.venv`.
After cleanup, exit the project shell or deactivate manually: a child process cannot reset the
parent shell's activation variables.
Recreate Python with `just setup-python`, native libraries and Rust with `just setup-conda`, and
build artifacts with `just build`. The host C/C++ toolchain, Conda and Cargo package caches,
uv-managed Python installations, and lockfiles are retained.

To clean all three at once, run `just clean` (`just clean-all` remains an alias).
Recreate them with `just setup`, `just activate`, and `just build`.

To build the complete project, run:

```bash
just build
```

To run all Rust and Python tests:

```bash
just test
```

For a build without the Rust display integration, use:

```bash
just build-headless
just test-headless
```

### Supported platforms

The source is designed for native builds on Windows x86-64, macOS x86-64 and ARM64, and Linux x86-64 and ARM64, including Jetson-class devices. Conda-forge provides OpenCV 5 packages for these targets. Each device builds its own native artifacts; binaries and Python wheels are not interchangeable between operating systems or CPU architectures.

The default build includes display support. Headless builds do not compile or link the Rust HighGUI integration and are intended for UAV deployment. Camera indices use the backend selected by OpenCV on the host. File input is the portable option for automated verification; physical camera and CSI/GStreamer availability must be verified on the target hardware.

## Usage

Run the following commands from `just activate`. The CLI supports loading from a configuration file, selecting a camera index, or injecting a test video.

The CLI reads [`config/default.yaml`](config/default.yaml) on every launch. Edit that file to
change the project's runtime defaults. To use another configuration as the base, pass
`--config-path`; explicit CLI arguments override values from the selected file:

```bash
just run --config-path config/custom.yaml --camera-index 1 --display
```

`--config` and `-c` remain aliases for `--config-path`. Relative configuration paths are resolved
from the current working directory; `just run` uses the project root. Missing or malformed files
produce an error before capture opens. Partial files inherit omitted settings from the shipped
defaults; unknown keys are errors. The same YAML schema supplies Rust and Python defaults,
embedded at build time. Python accepts `config_path="config/custom.yaml"`; explicit Python
arguments override that file. Without `config_path`, Python uses the embedded defaults and
does not depend on the current working directory.

The CLI opens a window only with `--display` or `--output-mode display`. A YAML
`output_mode: Display` without either flag is ignored with a warning in debug mode. Console logging
is controlled separately through `--mode none|info|debug` (default: `info`).

[`config/default.yaml`](config/default.yaml) lists every supported setting. YAML and Python use
snake_case names; CLI options use the same names with hyphens, such as `tau_min` / `--tau-min`.
Algorithm settings live under `algorithm_config` in YAML and directly on Rust's `AlgorithmConfig`.
The other settings belong to `AppConfig`.

| Settings | Purpose |
| --- | --- |
| `processing_type`, `input_resolution` | Algorithm selection (`motion`) and processing/mask dimensions from `config/default.yaml`. |
| `fb_max`, `ransac_h`, `max_corners`, `quality_level`, `min_distance`, `min_inliers` | Sparse feature selection and reliable background-motion estimation. |
| `parallax_threshold`, `tau_epi` | Parallax detection and epipolar residual threshold. |
| `flow_preset`, `finest_scale` | DIS quality preset (`ultrafast`, `fast`, `medium`) and finest pyramid level; zero retains full-resolution flow. |
| `tau_min`, `k_sigma` | Minimum motion threshold and adaptive noise multiplier. |
| `morphology_open_size`, `morphology_close_size`, `min_component_area` | Mask cleanup kernel sizes and minimum component area. |
| `search_score_threshold`, `tracking_score_threshold` | Candidate score thresholds before and after target confirmation. |
| `confirmation_window`, `confirmation_hits`, `tracker_gate_scale` | Confirmation history and association search area. |
| `max_coast_frames`, `min_correlation` | Maximum tracking time without motion (zero disables propagated masks) and minimum appearance agreement for observations and coasting. |
| `output_mode`, `display_resolution` | Headless/display output and optional preview dimensions. |
| `mode` | Console logging: `none`, `info`, or `debug`; independent of the display option. |
| `camera_index`, `video_path`, `resolution`, `fps` | Camera selection, optional file input, capture dimensions, and requested capture rate (30 FPS by default). |

Values are validated together after overrides, before capture opens. For example, confirmation
hits cannot exceed the history window, morphology sizes must be positive odd numbers, and
the finest flow level must fit the processing dimensions. `null` is reserved for optional choices:
no video file, native camera capture mode, and a preview matching the captured frame.

**Run with a live camera stream (with display window):**
```bash
just run --camera-index 0 --display
```

**Run headlessly with a live camera:**
```bash
just run --camera-index 0
```

**Run using a test video file:**
```bash
just run --video-path fixtures/gemini_generated_video_8813b1b0.mp4 --display
```

### Capture settings and algorithm resolution

`fps` requests the live camera's capture rate, defaulting to 30. For example:

```bash
just run --fps 60 --display
```

The selected camera mode must support that rate at the requested resolution. Startup
logs show the requested and backend-reported rate; rejection, unavailable reporting,
or a difference greater than 1% produces a warning in debug mode. Nominal 30/60 modes reporting
29.97/59.94 are accepted. Capture continues at the available rate if the request fails.
The application adds no FPS sleep, frame dropping, or software rate limit. This option
is ignored for video files, which continue processing as fast as the pipeline allows.
Logged and preview FPS measure end-to-end throughput, not just the requested rate.

`resolution` requests the camera capture size. Omit it to keep the camera's default mode.
The camera must support the requested size; rejection or a different delivered size is an error.
This option does not change the resolution of an input video file.

`input_resolution` resizes the captured image before passing it to the core algorithm, independently
of the camera size. Both upscaling and downscaling are supported. Returned masks
and bounding-box coordinates use the algorithm input dimensions. The display resizes the mask back
to the camera image using nearest-neighbour interpolation before applying the overlay.
The processing and returned mask size defaults to `algorithm_config.input_resolution` in
`config/default.yaml`. Explicit `input_resolution: null` also selects
the embedded processing default; rebuild Rust/Python after changing embedded defaults.
`display_resolution` independently resizes the final preview; it does not alter capture or results.
Changing the aspect ratio stretches the image; choose matching aspect ratios when that is unwanted.

CLI:

```bash
just run --camera-index 0 --resolution 1280x720 --input-resolution 640x360 --display
```

Python:

```python
diff_motion.run(
    camera_index=0,
    headless=False,
    resolution=(1280, 720),
    fps=60.0,
    input_resolution=(640, 360),
    tau_min=1.2,
)
```

YAML configuration (`just run --config-path config/custom.yaml --display`):

```yaml
algorithm_config:
  processing_type: motion
  input_resolution:
    width: 640
    height: 360
resolution:
  width: 1280
  height: 720
output_mode: Display
camera_index: 0
fps: 60.0
video_path: null
```

CLI values override the corresponding YAML fields. Resolution dimensions must be positive
and fit OpenCV's integer range. Algorithm input must have both dimensions at least 8 pixels and
at least one dimension at least 12 pixels, as required by optical flow. Python's resolution options are keyword-only; existing positional
arguments remain supported.

### Per-frame output API

Rust callers can receive each frame and its processing result with
`process_video_stream_with_callback`. The existing `process_video_stream` function remains
available for callers that do not need per-frame output.

```rust
use diff_motion_core::{
    config::AppConfig,
    process_video_stream_with_callback,
};

let config = AppConfig::default();
process_video_stream_with_callback(config, |frame, result| {
    println!(
        "{}x{} frame: {} detected objects",
        frame.width,
        frame.height,
        result.objects.len()
    );
    Ok(())
})?;
```

The callback receives `&Frame` and `&ProcessingResult` and returns `Result<(), String>`. It is
given the original captured frame; the result mask and bounding boxes
use the configured processing dimensions. It is
called synchronously for every processed frame, including the first frame, whose result contains
an all-zero mask and no detected objects because optical flow requires a previous frame. Returning an
error stops capture and returns that error to the caller.

Python's `run` function accepts an optional `callback` argument:

```python
import diff_motion


def handle_result(result):
    mask = result["mask"]
    print(mask["width"], mask["height"], len(result["objects"]))


diff_motion.run(headless=True, camera_index=0, callback=handle_result)
```

The callback is called synchronously for every processed frame, including the first empty result,
with a dictionary in this form:

```python
{
    "mask": {
        "data": bytes,  # Binary 0/1 values in row-major order.
        "width": int,
        "height": int,
    },
    "objects": [
        {
            "class_id": int,
            "class_name": str,
            "confidence": float,  # Raw residual score, not a probability.
            "bbox": {
                "x": float,
                "y": float,
                "width": float,
                "height": float,
            },
        }
    ],
}
```

Core video capture and processing release Python's global interpreter lock (GIL). The callback
reacquires the GIL while it runs. Callback exceptions propagate unchanged and stop capture. Keep
callback work short or hand data off to another worker, because synchronous callback execution
directly reduces processing throughput.

### Output modes and tracking states

Console logging uses the same `mode` setting in CLI, YAML, Rust, and Python:

| Mode | Console output |
| --- | --- |
| `none` | No application logs, including warnings/errors; failures still return errors or a nonzero exit code. |
| `info` | Startup/source messages, FPS, mask percentages, camera-motion reliability, and errors. This is the default. |
| `debug` | Everything in `info`, plus warnings, the configuration and per-frame processing time, dimensions, object count, foreground pixels, and camera-motion reliability. |

For example, use `just run --mode debug --display`, `mode: none` in YAML, or
`diff_motion.run(mode="none")` in Python. Explicit arguments override the selected YAML.
Each stream has its own logging mode; changing one call does not affect another.
OpenCV backends can print their own diagnostics directly to stderr independently of
this application logger. `info` avoids the per-frame console I/O added by `debug`.

Enabled logs write to stderr with `[INFO]`, `[WARN]`, `[ERROR]`, and `[DEBUG]`
levels. In `info` and `debug` modes, once per second the logger reports processed
FPS and the percentages of background (`mask_0`) and target (`mask_1`) pixels:

```text
[INFO] fps=59.84 mask_0=99.62% mask_1=0.38%
```

This is an example format, not a throughput measurement. FPS includes capture,
processing, callbacks, and display work. Pixel percentages are weighted across all
masks in that reporting interval, including the initial all-zero mask. A final summary
covers the whole stream, even for clips shorter than one second. Missing measurements
are shown as `n/a`.

Unreliable camera-motion estimates remain INFO summaries with the affected percentage;
the initialization frame is excluded. For example, `ego_motion_unreliable=3.33%` means
one in thirty estimates failed in that interval. New detections are paused only during
those estimates and resume when estimation succeeds; existing tracks may coast.
This reliability report does not cap FPS. Frame counts are omitted from console diagnostics.
Warnings appear only in debug mode, including a camera returning no frame,
while normal file EOF ends cleanly. With logging enabled, runtime errors are logged once and still propagate
to the caller. The preview shows a small FPS label for the last completed interval;
`FPS: --` appears until the first measurement is available. The preview FPS label remains
active in `none` mode; mask statistics are skipped when console logging is disabled.

`Headless` processes video without opening a window; Rust and Python callbacks can receive every
result. `Display` overlays the semantic mask on the camera image in red at 30% opacity,
while keeping callbacks available. Mask blending leaves background pixels unchanged; the preview
also includes the FPS label. The API mask remains binary 0/1.
Close the window or press `q`, `Q`, or Esc to stop capture and exit. OpenCV's native Wayland backend
does not report window closure; use the keyboard or a Qt/GTK backend there. Bounding boxes remain
available as metadata through the callbacks. A
headless build rejects display requests. `JsonCamera` is reserved and returns an explicit
unsupported-mode error. There is currently one optical-flow algorithm; `processing_type` does
accepts `motion` (or the legacy empty string); other algorithms are rejected. Use
`--display` or `--output-mode display` to enable the CLI window; `--output-mode headless`
keeps it off. Python supports `output_mode="display"` or the existing
`headless=False`, but these two overrides cannot be supplied together.

Tracking runs automatically in both output modes:

The table describes the shipped defaults; confirmation, association, and coasting settings above
can be changed through YAML, CLI, or Python.

| State | Behavior |
| --- | --- |
| SEARCH | Confirm a consistent motion candidate in at least three of the last five usable frames before publishing its mask. Invalid ego-motion estimates cannot start a detection. |
| LOCK | Predict an association gate with a Kalman filter and choose measured motion inside it. Require appearance agreement before refreshing the track. Camera-motion estimation excludes only the tracked box. |
| COAST | Match the saved grayscale template when motion disappears and transform the previous binary mask to the matched location. Reacquire motion or drop the track after `max_coast_frames` or correlation below `min_correlation`. |

During COAST, the mask is propagated from a previous motion observation. The object's confidence
retains its last measured residual score. Tracking cannot discover a previously unseen target
that has no relative motion.

For a deforming target such as a hand, a propagated silhouette can retain an old shape.
Set `max_coast_frames: 0` in YAML, `max_coast_frames=0` in Python, or run:

```bash
just run --display --input-resolution 256x256 --max-coast-frames 0
```

This clears the mask when no associated motion is observed. Kalman prediction still guides
association; it does not move a freshly measured mask. To use the same motion threshold inside
and outside the predicted gate, set `tracking_score_threshold` equal to `search_score_threshold`.
Current-frame score projection uses nearest pixels; expanding motion can leave small holes,
especially when morphology is disabled with kernel size 1.

### Algorithm verification

Four short integration tests use textured image data from the supplied generated video, known
camera translations with jitter, and an injected 40x28-pixel textured patch. These fixtures
explicitly process at 640×360 so their geometry and IoU checks stay reproducible when runtime
YAML defaults change; these test parameters never affect application configuration. They check false
foreground area, mask and box IoU of at least 0.5, and localization through three frames of zero
relative motion followed by reacquisition. A sharp direction-reversal test requires mask IoU
of at least 0.7 at the current target position. Tracker unit tests check confirmation, mask alignment,
expiry, and loss of an unrelated or disappeared target. Run them through `just test` or
`just test-headless`.

Degenerate epipolar fits can trigger OpenCV's `!model.empty()` assertion. That specific
failure is treated as an unavailable optional epipolar cue while retaining the valid
homography; unrelated OpenCV errors still propagate. Regression tests cover this failure,
valid fits, error propagation, and recovery after textureless sky and exposure changes.

Clear sky remains a detection limitation: without enough background features, camera motion
cannot be estimated reliably and new target acquisition is paused. Colour or local uniformity
could refine a motion-confirmed silhouette, but colour alone cannot identify a UAV or reliably
separate it from birds, clouds, and lens contamination. No appearance-based acquisition is enabled.

By default DIS uses the `fast` preset with `finest_scale: 0`, computing flow at the full working
resolution to resolve small targets. This increases
computation compared with the preset's coarse pyramid output; real-time throughput, parallax,
occlusion, scale changes, and camera backends still require evaluation on flight footage and
target hardware.

Residual scoring, mask thresholding, and larger display overlays use Rayon to process independent
rows in parallel. Smaller overlays use the serial path to avoid scheduling overhead. Frame order,
tracking updates, and adaptive noise-floor selection remain sequential. OpenCV also parallelizes
parts of optical flow through its own backend. `RAYON_NUM_THREADS` can limit Rayon's worker pool;
it does not control OpenCV's threads.

The residual-score microbenchmark compares the original serial calculation with one, four, and
the default number of Rayon workers. Run it from the activated project shell:

```bash
cargo test --release --package diff-motion-core benchmark_residual_score_parallelism -- --ignored --nocapture
```

On the development Mac, four-worker scoring was about 3–4 times faster at 640×360 and 1280×720,
with bitwise-identical scores. This does not imply the same speedup for the complete application:
three alternating full-video runs gave medians of 6.27 seconds before and 6.23 seconds after,
within run-to-run variation. A stage profile attributed about 82% of processing time to dense
flow and 12% to camera-motion estimation. Jetson, Linux, Windows, and live-camera performance
still require measurement.

The processing pipeline also reuses LK image pyramids across both tracking directions
and consecutive frames, reuses image/morphology buffers, and swaps grayscale frames
instead of copying them. Component scores are accumulated in one image pass rather
than scanning the full image for every component. Resolution, flow refinement settings,
and detection thresholds are unchanged.

Run the full processing benchmark from the activated project shell:

```bash
cargo test --release --package diff-motion-core --test performance -- --ignored --nocapture
```

It preloads the first 120 fixture frames, processes them three times, and reports mean,
median, p95 latency, throughput, and an output fingerprint. Decoding, display, and callback
time are excluded. On the development Mac, alternating baseline/optimized runs at
1280×720 capture and 640×360 processing measured approximately 26.3 ms versus 25.2 ms
per frame (38 versus 40 FPS). The mask, box, and confidence fingerprint was identical;
the tests also compare cached-pyramid LK results bit for bit with the original calls.

These measurements do not meet the 60 FPS target, which requires a total budget of
16.7 ms per frame. Native DIS flow averaged 21.5 ms across 357 processed frame pairs
in a separate stage-timing run, already exceeding that budget. Its preset, pyramid scale,
and refinement iterations were preserved to avoid lowering quality. Sustained 60 FPS
needs further native-flow acceleration verified on the deployment hardware, plus a
camera mode that actually supplies 60 FPS. Use `--fps 60` to request it; selecting a
capture resolution alone does not request that frame rate or guarantee the throughput.

## Algorithm Overview

The algorithm consists of the following per-frame pipeline stages:
1. **Ego-Motion (Sparse LK + RANSAC):** Computes background homography $H$ and, if significant parallax is detected, the fundamental matrix $F$.
2. **Aligned Dense Flow:** Warps the previous frame into the current frame's background coordinates using $H$, then computes residual optical flow toward the current frame.
3. **Residual Scoring:** Combines flow magnitude and epipolar distance, then relocates scores along the measured flow to the current image before building the mask.
4. **Morphology & Bounding Box:** Thresholds the score, applies morphology operations to clean up noise, and derives a bounding box from the resulting target blob.
5. **Tracking:** Confirms motion candidates, associates detections with the predicted target, and temporarily propagates its mask when relative motion disappears.

For a deep dive into the algorithm, see [docs/CORE_ALGHORITHM.md](docs/CORE_ALGHORITHM.md).

## Project Guidelines

For information on development priorities, testing rules, and conventions, please see [AGENTS.md](AGENTS.md).
