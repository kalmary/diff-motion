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

- **Rust** (Edition 2021)
- **Conda** (Miniconda, Anaconda, or Micromamba)
- **uv** (fast Python package installer and resolver, install via `curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`)
- **Just** (command runner, install via `cargo install just` or `brew install just`)

### Environment Setup

This project uses Conda to manage C++ dependencies (OpenCV, Clang, CMake), while Python libraries are managed via a `uv` virtual environment, isolating them from the global system.

To create the Conda environment and install all necessary dependencies:

```bash
just build-conda
```

*Note: This command will use `conda env update` to either create or update the `diff-motion` environment defined in `environment.yml`.*

## Building & Testing

To compile the Rust project, initialize the Python `uv` virtual environment, and build the Python bindings, run:

```bash
just build-project
```

To run the test suite (both Rust and Python tests):

```bash
just test
```

## Usage

First, activate the Conda environment:
```bash
conda activate diff-motion
```

You can then run the application via the CLI crate. The CLI supports loading from a configuration file, selecting a camera index, or injecting a test video.

**Run with a live camera stream (with display window):**
```bash
cargo run --release --bin diff-motion-cli -- --camera-index 0 --display
```

**Run headlessly with a live camera:**
```bash
cargo run --release --bin diff-motion-cli -- --camera-index 0
```

**Run using a test video file:**
```bash
cargo run --release --bin diff-motion-cli -- --video-path tests/fixtures/sample_flight.mp4 --display
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
