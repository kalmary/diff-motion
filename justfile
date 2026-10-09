[unix]
set shell := ["sh", "-cu"]

[windows]
set shell := ["powershell.exe", "-NoLogo", "-Command"]

# List the available project workflows.
default:
    @just --list

# Prepare the Conda, Rust, and Python development environments.
setup: setup-conda setup-rust setup-python
    @echo "Setup complete. Activate the environment with: conda activate diff-motion"

# Create or update the Conda environment and its native dependencies.
setup-conda:
    conda env update --file environment.yml --prune

# Install the required Rust components and fetch locked Cargo dependencies.
setup-rust:
    rustup toolchain install stable --component rustfmt --component clippy
    cargo +stable fetch --locked

# Install the recorded Python development and test tools in the Conda environment.
setup-python: setup-conda
    conda run --no-capture-output --name diff-motion uv pip install --system --group dev

# Open a project shell with the Conda environment and stable Rust toolchain active.
[unix]
activate:
    #!/bin/sh
    exec conda run --no-capture-output --name diff-motion rustup run stable "${SHELL:-/bin/sh}" -i

[windows]
activate:
    conda run --no-capture-output --name diff-motion rustup run stable powershell.exe -NoLogo -NoExit

# Open a shell inside the project Conda environment.
[unix]
activate-conda:
    #!/bin/sh
    exec conda run --no-capture-output --name diff-motion "${SHELL:-/bin/sh}" -i

[windows]
activate-conda:
    conda run --no-capture-output --name diff-motion powershell.exe -NoLogo -NoExit

# Open a shell using the stable Rust toolchain.
[unix]
activate-rust:
    #!/bin/sh
    exec rustup run stable "${SHELL:-/bin/sh}" -i

[windows]
activate-rust:
    rustup run stable powershell.exe -NoLogo -NoExit

# Start Python inside the project Conda environment.
activate-python:
    conda run --no-capture-output --name diff-motion python

# Build the Rust application and Python extension.
build: _require-conda _build-rust _build-python

# Build the Rust application and Python extension without display support.
build-headless: _require-conda _build-rust-headless _build-python-headless

# Run the Rust and Python test suites.
test: _require-conda _test-rust _test-python

# Run the Rust and Python test suites without display support.
test-headless: _require-conda _test-rust-headless _test-python-headless

# Verify formatting, linting, builds, and tests with and without display support.
verify: _require-conda _format-check _clippy _clippy-headless build test test-headless

# Run the CLI inside the active Conda environment.
run *args: _require-conda
    cargo run --release --package diff-motion-cli -- {{ args }}

[private]
_require-conda:
    python -c "import os; active = os.environ.get('CONDA_DEFAULT_ENV'); assert active == 'diff-motion', f'Activate the diff-motion Conda environment first (active: {active})'"

[private]
_build-rust:
    cargo build --release --package diff-motion-cli

[private]
_build-rust-headless:
    cargo build --release --package diff-motion-cli --no-default-features

[private]
_build-python:
    maturin develop --release

[private]
_build-python-headless:
    maturin develop --release --no-default-features

[private]
_test-rust:
    cargo test --workspace --all-features

[private]
_test-rust-headless:
    cargo test --workspace --no-default-features

[private]
_test-python: _build-python
    pytest tests

[private]
_test-python-headless: _build-python-headless
    pytest tests

[private]
_format-check:
    cargo fmt --all -- --check

[private]
_clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

[private]
_clippy-headless:
    cargo clippy --workspace --all-targets --no-default-features -- -D warnings
