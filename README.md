# Diff Motion

**Diff Motion** is a real-time computer-vision application designed to run on an NVIDIA Jetson Nano mounted on an unmanned aerial vehicle (UAV). It detects a target UAV purely based on relative motion discrepancy (optical flow separation) compared to the background. 

This project uses a **strictly deterministic algorithm** based on ego-motion compensation, dense optical flow, and epipolar residual testing. **No neural networks, deep learning models, or AI are used for the segmentation.**

## Features

- **Purely Deterministic:** No AI models. Relies entirely on classical computer vision (Optical Flow, RANSAC, Epipolar geometry).
- **Headless & Display Modes:** Can run entirely headlessly on the Jetson Nano, or optionally open a display window showing the semantic mask and bounding boxes overlaid on the live camera stream.
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
| `just setup-conda` | Creates or updates `diff-motion` with Rust 1.99.0 and the native OpenCV/build dependencies. |
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
Routine build and test workflows do not provision Conda dependencies;
their locked uv sync rebuilds and reinstalls the Python extension in `.venv`.

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

`Headless` processes video without opening a window; Rust and Python callbacks can receive every
result. `Display` adds the video window and overlays while keeping callbacks available. A
headless build rejects display requests. `JsonCamera` is reserved and returns an explicit
unsupported-mode error. There is currently one optical-flow algorithm; `processing_type` does
not yet select an alternative processor.

Tracking runs automatically in both output modes:

| State | Behavior |
| --- | --- |
| SEARCH | Select a motion candidate; confirm a consistent target in at least three of the last five usable frames. Invalid ego-motion estimates cannot start a detection. |
| LOCK | Predict center and velocity with a Kalman filter, retain the measured box size, and associate nearby motion components. Exclude the target region from camera-motion estimation. |
| COAST | Match the saved grayscale template when motion disappears and transform the previous binary mask to the matched location. Reacquire motion or drop the track after 15 frames or correlation below 0.5. |

During COAST, the mask is propagated from a previous motion observation. The object's confidence
retains its last measured residual score. Tracking cannot discover a previously unseen target
that has no relative motion.

### Algorithm verification

Three short integration tests use textured image data from the supplied generated video, known
camera translations with jitter, and an injected 40x28-pixel textured patch. They check false
foreground area, mask and box IoU of at least 0.5, and localization through three frames of zero
relative motion followed by reacquisition. Tracker unit tests check confirmation, mask alignment,
expiry, and loss of an unrelated or disappeared target. Run them through `just test` or
`just test-headless`.

DIS flow is computed at the full working resolution to resolve small targets. This increases
computation compared with the preset's coarse pyramid output; real-time throughput, parallax,
occlusion, scale changes, and camera backends still require evaluation on flight footage and
target hardware.

## Algorithm Overview

The algorithm consists of the following per-frame pipeline stages:
1. **Ego-Motion (Sparse LK + RANSAC):** Computes background homography $H$ and, if significant parallax is detected, the fundamental matrix $F$.
2. **Aligned Dense Flow:** Warps the current frame using $H$ to align it with the previous frame, and computes the dense optical flow of the residual movement.
3. **Residual Scoring:** Combines the optical flow magnitude and the Sampson distance to the epipolar lines to score moving objects out of the background.
4. **Morphology & Bounding Box:** Thresholds the score, applies morphology operations to clean up noise, and derives a bounding box from the resulting target blob.
5. **Tracking:** Confirms motion candidates, associates detections with the predicted target, and temporarily propagates its mask when relative motion disappears.

For a deep dive into the algorithm, see [docs/CORE_ALGHORITHM.md](docs/CORE_ALGHORITHM.md).

## Project Guidelines

For information on development priorities, testing rules, and conventions, please see [AGENTS.md](AGENTS.md).
