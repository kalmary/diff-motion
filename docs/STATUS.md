# Status

## Work Log

- Updated `AGENTS.md` and `docs/PROJECT_GOAL.md` to specify that the goal is to determine a semantic mask based on observations, specifically outputting a binary mask where the UAV is 1 and the background is 0.
- Renamed `docs/status.md` to `docs/STATUS.md` and updated `AGENTS.md` to reflect the capitalized filename.
- Read all markdown files in the repository (AGENTS.md, README.md, docs/PROJECT_GOAL.md, docs/PLAN.md, docs/STATUS.md) to build context on project architecture and entry points.
- Restructured repository to use Cargo workspace with separate crates for core, cli, and python wrapper. Added testing guidelines for Rust and Python, added .cargo/config.toml for macOS compilation, and initialized basic tests.
- Defined abstract  trait and core data structures (, , , etc.) to support agnostic implementation of target detection algorithms as requested.
- Defined abstract FrameProcessor trait and core data structures (Frame, BoundingBox, SemanticMask, etc.) to support agnostic implementation of target detection algorithms as requested.
- Updated Python bindings via PyO3 to accept configuration parameters, specifically supporting `headless=True` mode out of the box, and added pytest verification tests.
- Drafted a comprehensive prompt defining the dynamic motion segmentation problem for the user to consult external algorithmic models.
- Copied the algorithm generation prompt to `docs/algorithm_prompt.md` as requested.
- Removed all YOLO/object detector assumptions from the architecture and updated `AGENTS.md`, `docs/algorithm_prompt.md`, and configurations to reflect pure motion-based target isolation.
- Updated `AGENTS.md`, `docs/PROJECT_GOAL.md`, and `docs/algorithm_prompt.md` to explicitly mandate that the solution must be a purely deterministic algorithm based entirely on optical flow separations, strictly forbidding neural networks.
- Updated `.gitignore` to include standard Rust ignore patterns (e.g. `target/` directory).
- Added `video_path` configuration argument (default: None) to the CLI (`--video-path`) and Python bindings to allow injecting test videos in place of the live camera stream.
- Read `docs/CORE_ALGHORITHM.md` to understand the deterministic ego-motion and epipolar flow algorithm. Added `opencv` crate to `core` to begin implementing the pipeline.
- Implemented initialization and Stage 1 (Ego-motion via Sparse LK + RANSAC) of the optical flow segmentation pipeline into `crates/core/src/processor.rs`.
- Resolved OpenCV 5.0 compilation issues on Mac by updating `opencv` rust bindings crate to 0.101.0 and updating module imports (`features` and `geometry`).
- Implemented Stage 2 (Aligned Dense Flow), Stage 3 (Epipolar Residual Scoring), and Stage 4 (Morphology & Bounding Box) in `OpticalFlowProcessor`.
- Implemented `process_video_stream` in `lib.rs` to run the main capture loop and optionally display the video stream with the overlaid semantic mask and bounding box.
- Wrote initial `README.md` detailing project goals, architecture, prerequisites, build instructions, and usage.
- Refactored environment configuration to restrict Conda to C++ dependencies and system packages, managing Python explicitly via a `uv` virtual environment, and updated `environment.yml`, `justfile`, and `README.md` accordingly.
- Removed unnecessary runner commands from `justfile` and updated `README.md` usage instructions.
- Removed the `clean` task from `justfile` as native `cargo clean` and manual deletions are preferred.
