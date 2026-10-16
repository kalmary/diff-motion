[unix]
set shell := ["sh", "-cu"]

[windows]
set shell := ["powershell.exe", "-NoLogo", "-Command"]

# List the available project workflows.
default:
    @just --list

# Prepare the Conda-managed native/Rust tools and uv Python environment.
setup: setup-conda setup-rust setup-python
    @echo "Setup complete. Open the project shell with: just activate"

# Create or update the Conda environment with native dependencies and pinned Rust.
setup-conda:
    conda env update --file environment.yml --prune
    conda run --no-capture-output --name diff-motion cargo clean --package opencv
    conda run --no-capture-output --name diff-motion cargo clean --release --package opencv

# Prepare and check the pinned Conda Rust tools and fetch locked Cargo dependencies.
setup-rust: setup-conda
    conda run --no-capture-output --name diff-motion cargo --version
    conda run --no-capture-output --name diff-motion rustc --version
    conda run --no-capture-output --name diff-motion cargo fmt --version
    conda run --no-capture-output --name diff-motion cargo clippy --version
    conda run --no-capture-output --name diff-motion cargo fetch --locked

# Create and sync the uv-managed Python virtual environment and development/test tools.
setup-python:
    uv sync --managed-python --locked --no-install-project --group test

# Clear uv's shared cache and remove the project's Python virtual environment.
[unix]
clean-python:
    uv cache clean
    rm -rf -- .venv

# Clear uv's shared cache and remove the project's Python virtual environment.
[windows]
clean-python:
    uv cache clean; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (Test-Path -LiteralPath .venv) { Remove-Item -LiteralPath .venv -Recurse -Force -ErrorAction Stop }

# Remove the project's Rust/native Conda environment from the base manager context.
[unix]
clean-conda:
    #!/bin/sh
    set -eu
    conda_environments=$(conda env list)
    if printf '%s\n' "$conda_environments" | awk '$1 == "diff-motion" { found = 1 } END { exit !found }'; then
        conda run --no-capture-output --name base conda env remove --name diff-motion --yes
    fi

# Remove the project's Rust/native Conda environment from the base manager context.
[windows]
clean-conda:
    $environmentJson = conda env list --json; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; $projectEnvironment = ($environmentJson | ConvertFrom-Json -ErrorAction Stop).envs | Where-Object { (Split-Path -Leaf $_) -eq "diff-motion" }; if ($projectEnvironment) { conda run --no-capture-output --name base conda env remove --name diff-motion --yes; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

# Remove Cargo build artifacts for the workspace.
[unix]
clean-rust:
    #!/bin/sh
    set -eu
    conda_environments=$(conda env list)
    if printf '%s\n' "$conda_environments" | awk '$1 == "diff-motion" { found = 1 } END { exit !found }'; then
        conda run --no-capture-output --name diff-motion cargo clean
    else
        rm -rf -- target
    fi

# Remove Cargo build artifacts for the workspace.
[windows]
clean-rust:
    $environmentJson = conda env list --json; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; $projectEnvironment = ($environmentJson | ConvertFrom-Json -ErrorAction Stop).envs | Where-Object { (Split-Path -Leaf $_) -eq "diff-motion" }; if ($projectEnvironment) { conda run --no-capture-output --name diff-motion cargo clean; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } elseif (Test-Path -LiteralPath target) { Remove-Item -LiteralPath target -Recurse -Force -ErrorAction Stop }

# Clean Cargo artifacts, uv's shared cache, .venv, and the project Conda environment.
clean: clean-rust clean-python clean-conda

# Alias for complete project cleanup.
clean-all: clean

# Open a project shell with pinned Conda Rust/native tools and uv Python active.
[unix]
activate:
    #!/bin/sh
    exec conda run --no-capture-output --name diff-motion uv run --no-sync -- "${SHELL:-/bin/sh}" -i

# Open a project shell with pinned Conda Rust/native tools and uv Python active.
[windows]
activate:
    conda run --no-capture-output --name diff-motion uv run --no-sync -- powershell.exe -NoLogo -NoExit

# Open a shell with the project's pinned Rust and native Conda dependencies active.
[unix]
activate-conda:
    #!/bin/sh
    exec conda run --no-capture-output --name diff-motion "${SHELL:-/bin/sh}" -i

# Open a shell with the project's pinned Rust and native Conda dependencies active.
[windows]
activate-conda:
    conda run --no-capture-output --name diff-motion powershell.exe -NoLogo -NoExit

# Open a shell with the uv-managed Python virtual environment active.
[unix]
activate-python:
    #!/bin/sh
    exec uv run --no-sync -- "${SHELL:-/bin/sh}" -i

# Open a shell with the uv-managed Python virtual environment active.
[windows]
activate-python:
    uv run --no-sync -- powershell.exe -NoLogo -NoExit

# Build the Rust application and Python extension.
build: _require-conda _require-rust _require-python _build-rust _build-python

# Build the Rust application and Python extension without display support.
build-headless: _require-conda _require-rust _require-python _build-rust-headless _build-python-headless

# Run the Rust and Python test suites.
test: _require-conda _require-rust _require-python _test-rust _test-python

# Run the Rust and Python test suites without display support.
test-headless: _require-conda _require-rust _require-python _test-rust-headless _test-python-headless

# Verify formatting, linting, builds, and tests with and without display support.
verify: _require-conda _require-rust _require-python _format-check _clippy _clippy-headless build test build-headless test-headless

# Run the Rust CLI with the native Conda dependencies active.
run *args: _require-conda _require-rust
    cargo run --release --package diff-motion-cli -- {{ args }}

[private]
[unix]
_require-conda:
    @test "${CONDA_DEFAULT_ENV:-}" = diff-motion || { echo "Native Conda environment is inactive. Run just activate." >&2; exit 1; }

[private]
[windows]
_require-conda:
    @if ($env:CONDA_DEFAULT_ENV -ne "diff-motion") { throw "Native Conda environment is inactive. Run just activate." }

[private]
[unix]
_require-rust:
    @for rust_tool in cargo rustc; do case "$(command -v "$rust_tool")" in "$CONDA_PREFIX"/*) ;; *) echo "$rust_tool must come from the project Conda environment. Run just activate." >&2; exit 1 ;; esac; done

[private]
[windows]
_require-rust:
    @foreach ($rustTool in @("cargo", "rustc")) { $toolPath = (Get-Command $rustTool -ErrorAction Stop).Source; $prefix = [System.IO.Path]::GetFullPath($env:CONDA_PREFIX) + [System.IO.Path]::DirectorySeparatorChar; if (-not $toolPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) { throw "$rustTool must come from the project Conda environment. Run just activate." } }

[private]
[unix]
_require-python:
    @test -n "${VIRTUAL_ENV:-}" || { echo "Python virtual environment is inactive. Run just activate." >&2; exit 1; }
    @command -v python >/dev/null || { echo "Python virtual environment is inactive. Run just activate." >&2; exit 1; }
    @python -c "import pathlib, sys; expected = pathlib.Path('.venv').resolve(); assert pathlib.Path(sys.prefix).resolve() == expected, 'The uv virtual environment is inactive. Run just activate.'"

[private]
[windows]
_require-python:
    @if (-not $env:VIRTUAL_ENV) { throw "Python virtual environment is inactive. Run just activate." }
    @if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python virtual environment is inactive. Run just activate." }
    @python -c "import pathlib, sys; expected = pathlib.Path('.venv').resolve(); assert pathlib.Path(sys.prefix).resolve() == expected, 'The uv virtual environment is inactive. Run just activate.'"

[private]
_build-rust:
    cargo build --release --package diff-motion-cli

[private]
_build-rust-headless:
    cargo build --release --package diff-motion-cli --no-default-features

[private]
_build-python:
    uv sync --managed-python --locked --group test --no-build-isolation --reinstall-package diff-motion --config-setting "maturin.build-args=--profile release"

[private]
_build-python-headless:
    uv sync --managed-python --locked --group test --no-build-isolation --reinstall-package diff-motion --config-setting "maturin.build-args=--profile release --no-default-features"

[private]
_test-rust:
    cargo test --workspace --all-features

[private]
_test-rust-headless:
    cargo test --workspace --no-default-features

[private]
_test-python: _build-python
    python -m pytest tests

[private]
_test-python-headless: _build-python-headless
    python -m pytest tests

[private]
_format-check:
    cargo fmt --all -- --check

[private]
_clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

[private]
_clippy-headless:
    cargo clippy --workspace --all-targets --no-default-features -- -D warnings
