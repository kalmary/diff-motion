set shell := ["bash", "-c"]

# Show available commands
default:
    @just --list

# Create or update the Conda environment (C++ and Python system libraries)
build-conda:
    conda env update -f environment.yml --prune

# Build the Conda environment, then compile the Rust project and set up the Python venv using uv
build-project: build-conda
    conda run -n diff-motion cargo build --release
    conda run -n diff-motion uv venv .venv
    conda run -n diff-motion uv pip install maturin pytest
    conda run -n diff-motion uv run maturin develop -m crates/python/Cargo.toml --release

# Run the test suite (Rust and Python)
test:
    conda run -n diff-motion cargo test --workspace
    conda run -n diff-motion uv run pytest crates/python/tests/
