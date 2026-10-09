# Diff Motion

**Diff Motion** is a real-time computer-vision application designed to run on an NVIDIA Jetson Nano mounted on an unmanned aerial vehicle (UAV). It detects a target UAV purely based on relative motion discrepancy (optical flow separation) compared to the background. 

This project uses a **strictly deterministic algorithm** based on ego-motion compensation, dense optical flow, and epipolar residual testing. **No neural networks, deep learning models, or AI are used for the segmentation.**

## Features

- **Purely Deterministic:** No AI models. Relies entirely on classical computer vision (Optical Flow, RANSAC, Epipolar geometry).
- **Headless & Display Modes:** Can run entirely headlessly on the Jetson Nano, or optionally open a display window showing the semantic mask and bounding boxes overlaid on the live camera stream.
- **Cross-Platform & Highly Performant:** Written in Rust, leveraging OpenCV 5.0 C++ bindings for maximum performance.
- **Python Bindings:** Includes Python bindings via PyO3 for integration with Python-based drone control systems.

## Architecture

The project is structured as a Cargo workspace with three crates:

- **`crates/core` (`diff-motion-core`)**: The core algorithm library. Handles all OpenCV logic, image warping, homography, optical flow, and mask generation.
- **`crates/cli` (`diff-motion-cli`)**: The command-line entry point. Used for running the pipeline standalone.
- **`crates/python` (`diff-motion-python`)**: Python bindings via PyO3/Maturin. Allows importing the library as `import diff_motion` in Python.

## Prerequisites

- **Rust 1.88 or newer** (Edition 2021)
- **Conda** (Miniconda or Anaconda)
- **System Clang/LLVM** (e.g., Xcode Command Line Tools on macOS, Visual Studio Build Tools and LLVM on Windows, or `libclang-dev` on Linux)
- **uv** (fast Python package installer and resolver; use the installation method supported by your operating system)
- **Just** (cross-platform command runner; install with `cargo install just` or your system package manager)

### Environment Setup

This project uses one active Conda environment for Python and native dependencies such as OpenCV 5 and CMake. Rust is managed by the system Rust toolchain. `uv` installs the recorded Python development tools into the active Conda environment.

Prepare the Conda environment, Rust components, and Python development tools:

```bash
just setup
```

`setup` creates or updates the `diff-motion` Conda environment, installs the Rust formatting and linting components, fetches locked Cargo dependencies, and installs the recorded Python development tools, including `pytest`, into that environment. The individual stages can also be run separately:

```bash
just setup-conda
just setup-rust
just setup-python
```

The setup process runs Python installation commands inside `diff-motion`, even when the environment is not active. A child process cannot activate Conda in its parent shell, so activate it after setup before using the routine workflows:

```bash
conda activate diff-motion
```

Alternatively, open a child shell or interpreter prepared by Just:

```bash
just activate
```

The activation workflows are also available separately:

```bash
just activate-conda
just activate-rust
just activate-python
```

`activate` opens an interactive project shell with both the Conda environment and stable Rust toolchain selected. `activate-conda` and `activate-rust` open shells for their respective environments, while `activate-python` starts the project Python interpreter. Exit a child shell or interpreter to return to the original terminal.

Run setup explicitly after changing `environment.yml`, `pyproject.toml`, or the required Rust components. Routine build and test workflows do not update the environment.

## Building & Testing

The public Just recipes group operations across Rust, Python, and native dependencies:

| Command | Description |
| --- | --- |
| `just setup` | Prepare the Conda, Rust, and Python development environments. |
| `just setup-conda` | Create or update the Conda environment and native dependencies. |
| `just setup-rust` | Install Rust components and fetch locked Cargo dependencies. |
| `just setup-python` | Install Python development and test tools into the Conda environment. |
| `just activate` | Open a project shell with Conda and stable Rust active. |
| `just activate-conda` | Open a shell inside the project Conda environment. |
| `just activate-rust` | Open a shell using the stable Rust toolchain. |
| `just activate-python` | Start Python inside the project Conda environment. |
| `just build` | Build the Rust application and Python extension. |
| `just build-headless` | Build both entry points without OpenCV display support. |
| `just test` | Build the Python extension and run the Rust and Python test suites. |
| `just test-headless` | Run both test suites without OpenCV display support. |
| `just verify` | Verify formatting, linting, builds, and tests with and without display support. |
| `just run <arguments>` | Run the CLI with the active Conda environment. |

To build the complete project, run:

```bash
just build
```

To run all Rust and Python tests:

```bash
just test
```

For a deployment without GUI libraries, use:

```bash
just build-headless
just test-headless
```

### Supported platforms

The source is designed for native builds on Windows x86-64, macOS x86-64 and ARM64, and Linux x86-64 and ARM64, including Jetson-class devices. Conda-forge provides OpenCV 5 packages for these targets. Each device builds its own native artifacts; binaries and Python wheels are not interchangeable between operating systems or CPU architectures.

The default build includes display support. Headless builds do not compile or link the Rust HighGUI integration and are intended for UAV deployment. Camera indices use the backend selected by OpenCV on the host. File input is the portable option for automated verification; physical camera and CSI/GStreamer availability must be verified on the target hardware.

## Usage

Run the following commands after activating the `diff-motion` Conda environment. The CLI supports loading from a configuration file, selecting a camera index, or injecting a test video.

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

## Algorithm Overview

The algorithm consists of the following per-frame pipeline stages:
1. **Ego-Motion (Sparse LK + RANSAC):** Computes background homography $H$ and, if significant parallax is detected, the fundamental matrix $F$.
2. **Aligned Dense Flow:** Warps the current frame using $H$ to align it with the previous frame, and computes the dense optical flow of the residual movement.
3. **Residual Scoring:** Combines the optical flow magnitude and the Sampson distance to the epipolar lines to score moving objects out of the background.
4. **Morphology & Bounding Box:** Thresholds the score, applies morphology operations to clean up noise, and derives a bounding box from the resulting target blob.

For a deep dive into the algorithm, see [docs/CORE_ALGHORITHM.md](docs/CORE_ALGHORITHM.md).

## Project Guidelines

For information on development priorities, testing rules, and conventions, please see [AGENTS.md](AGENTS.md).
