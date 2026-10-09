# TASK

Build a real-time computer-vision application for an NVIDIA Jetson Nano mounted on an unmanned aerial vehicle (UAV). The application must:

- Capture a live video stream from a camera connected to the Jetson Nano.
- Detect the target UAV using a **purely deterministic algorithm based on optical flow separations**. No neural networks, deep learning models, or AI are permitted for the segmentation.
- Produce a semantic mask (binary mask where UAV is 1 and background is 0) and derive bounding boxes for every processed frame based entirely on this motion segmentation.
- Optionally display the live camera stream with bounding boxes and semantic mask overlaid on the image.
- Expose runtime configuration through a command-line interface, including:
  - processing type; initially, motion-based segmentation and bounding boxes are supported;
  - whether the processed video window is displayed;
  - display dimensions.

The core capture and inference pipeline must work without the optional display so that the application can run headlessly on the UAV.


# Engineering Guidelines

## Priorities

Produce the smallest correct change that satisfies the request. Optimize for correctness, clarity, maintainability, and compatibility with existing behavior. Every changed line must have a reason connected to the task.

When instructions conflict, follow this order:

1. The user's explicit request.
2. Repository-specific conventions and documented requirements.
3. The guidelines in this file.

## Understand Before Editing

- Read the relevant code, tests, configuration, and documentation before proposing a change.
- Trace the current behavior and identify the actual cause of a problem. Do not patch symptoms without understanding them.
- State assumptions when the repository does not provide enough evidence. Ask before making a choice that would materially change behavior or scope.
- For multi-step work, use a short plan in which each step has a concrete verification method.

## Design Data First and Design Guidelines

- Start with the data model, state transitions, inputs, outputs, ownership, and invariants.
- Prefer data structures that make valid states and the common path easy to express.
- Avoid accumulating conditionals to compensate for a poor representation. Improve the representation when doing so is within scope.
- Keep data flow explicit. Avoid hidden mutation, unnecessary shared state, and unclear ownership.
- Use classes, inheritance, patterns, or framework layers only when they provide a concrete benefit over simpler structures and functions.
- When choosing a way of achieving a task:
  - If a task is short and simple (no need to add weight to environment), code the solution directly.
  - If a task is more complex or requires a framework abstraction, use the framework layer or pattern.
  - If the task is complex and ready solution exists, use it.
  - Always pick library which is most popular and well-maintained choice in given field.
  - Keep modules separate and with layers which will make adding new features easier - another model choice, free space for data interpretation, interpreted data sending.

## Prefer Simple, Direct Code

- Write the simplest implementation that is clearly correct.
- Do not add speculative abstractions, extension points, configuration, or error handling for unsupported hypothetical cases.
- Avoid clever code when ordinary control flow communicates the behavior more clearly.
- Reuse an existing abstraction when it fits. Do not force a task into an abstraction that makes the common case harder to understand.
- Choose descriptive, general names based on purpose. Do not rename unrelated identifiers or over-elaborate names.
- Add comments only when they explain intent, constraints, or a non-obvious decision. Do not comment that merely restate the code. In general safe commenting code is best. Proper naming of its parts should usually suffice, unless there is a fancy solution, hard to understand without comments.
- Do not add references to AI tools, generated code, prompts, or the development conversation.
- When writing rust code i want separate modules to be in structure: module_name.rs + modelune_name/ - no mod.rs

## Keep Changes Surgical

- Change only files and lines required for the requested outcome.
- Match the repository's existing style, architecture, formatting, and dependency choices.
- Do not perform drive-by refactors, formatting sweeps, dependency upgrades, or unrelated cleanup.
- Remove imports, variables, helpers, and files made unused by the current change. Leave pre-existing dead code alone unless the user asks to address it.
- If an unrelated defect is discovered, report it separately instead of expanding the patch.
- Do not modify generated or vendored files unless the project workflow explicitly requires it.

## Preserve Existing Contracts

- Treat public APIs, file formats, configuration, command output, persisted data, and established workflows as contracts.
- Preserve backward compatibility unless the user explicitly requests a breaking change and accepts its consequences.
- Before changing a script or one of its functions, read the corresponding file documentation in `docs/documentation/` and treat its documented behavior as a requirement.
- When changing a script or function, preserve its input data and output exactly, including accepted formats, schemas, types, names, ordering, and externally observable behavior, unless the user explicitly requests a contract change.
- Consider boundary cases, malformed input, partial failure, retries, concurrency, and resource cleanup when they are relevant to the actual system.
- Never hide failures with broad exception handling, silent fallbacks, arbitrary retries, or default values that obscure invalid state.
- Do not weaken validation, authorization, security controls, or data integrity to make a test pass.

## Performance Must Be Evidence-Based

- Prioritize correctness and clarity unless performance is a stated requirement or a measured problem.
- Do not claim an optimization without a representative benchmark, profile, or other reproducible evidence.
- Consider algorithmic complexity, allocations, I/O, memory locality, and contention where they materially affect the workload.
- Keep hot paths direct, but do not trade readability for hypothetical gains.

## Verification

- Define success in observable, testable terms before declaring the work complete.
- Run the narrowest relevant checks first, then broader checks when the risk justifies them.
- Add or update tests for changed behavior and important regressions when a test suite exists.
- Test externally visible behavior rather than implementation details where practical.
- Report exactly what was verified. If a check could not be run, explain why and identify the remaining uncertainty.
- Do not describe work as complete when it is known to be broken or unverified.

## Review and Communication

- Review the final change for correctness, scope, regressions, security, and accidental artifacts.
- Support technical conclusions with code references, test output, measurements, or reproducible behavior.
- Be direct and specific about flaws in code or design, but remain professional and never attack a person.
- Distinguish confirmed facts from assumptions and recommendations.
- Rethink if found problem is a valid one, which may in fact affect the codebase.
- Explain the outcome and important tradeoffs concisely; avoid narrating routine tool use.

## Version Control

- Never push, pull, switch branches, merge, rebase, reset, stage files, or run any other Git operation without the user's explicit approval.
- Read-only access to `.git` does not grant permission to run Git commands.
- Do not leave tool-specific metadata, temporary files, debug output, or other traces unrelated to the requested result in the repository.

## Testing code

- We follow Test-Driven Development (TDD). Write the test first, see it fail, then write minimal code to pass.
- For Rust code (`crates/core`, `crates/cli`, `crates/python`):
  - Write unit tests in their original files (using `#[cfg(test)]`).
  - Write integration tests in the `tests/` directory within each crate.
  - Run tests with `cargo test` and ensure they pass.
- For Python code (wrappers, scripts, or integration tests):
  - Use `pytest` for testing.
- Always run tests once new code is written. If tests fail, explain why and identify the remaining uncertainty.

## Python dependencies

- Classify every Python package before adding it.
- Packages imported or otherwise required by application code belong in the main project dependencies.
- Packages used only for tests, coverage, linting, formatting, or other development verification belong in the `test` dependency group.
- Build backends and packaging requirements belong in `build-system`, not in the main or test dependency groups.
- Add packages with `uv`; do not install an unrecorded dependency directly into the environment.

## System Dependencies & Toolchains

- **DO NOT** attempt to provision or manage foundational compiler toolchains (like `rust`, `clang`, or `libclang`) via Conda unless specifically asked. Rely on the user's globally installed system dependencies (e.g., `cargo` via `rustup`, and system LLVM/Clang via macOS Xcode Command Line Tools or Linux package managers).
- **DO NOT** inject environment variables like `LIBCLANG_PATH` or `DYLD_FALLBACK_LIBRARY_PATH` into `justfile` recipes, shell commands, or `.cargo/config.toml` in an attempt to forcefully link macOS libraries or fix `dyld` errors related to Conda's isolated libraries.
- If a Rust C-bindings crate (like `opencv` / `clang-sys`) complains about missing `libclang` in Conda, resolve it by ensuring Conda is NOT providing `clang`, forcing the build scripts to naturally fall back to the system's properly-linked host compiler.

## Editing documentation

- As an LLM you are able to edit documentation when requested.
- You should do this withour use of extra python/ any other extra scripts.
- You can use script when file format doesn't support editing directly.
- After every prompt concerning repository work, add one concise bullet to the `Work Log` in `docs/STATUS.md`. Each bullet must summarize the request and the completed result.
- Keep at most 50 `Work Log` bullets. When adding a fifty-first entry, remove the oldest entry.

## Agent based workflow

- When task is complex and requires multiple steps, use an agent-based workflow.
- You can use Superpowers skills then.
- Selects each agent's tier, model, and reasoning effort according to its assigned task.
- Never uses Sol 6 or Sol 6.1 for delegated work.
- Uses Sol 5.6 with high reasoning for work orchestration, architecture, solution planning, ambiguous decisions, security analysis, difficult reviews, and designing test cases.
- Uses Terra 6 by default for well-specified coding tasks, implementation, repository exploration, routine testing, and documentation.
- Reserves Luna for mechanical, repetitive, low-risk, fully specified work—not substantive implementation.
- Requires explicitly selecting both the model and reasoning effort.
- Allows parallel agents only for independent tasks without overlapping file ownership.
- Requires every delegation to provide detailed context, exact scope, constraints, owned files, expected output, test cases, verification commands, and completion criteria.
- Keeps integration decisions and final verification with the primary agent.
- Requires reviewing and testing agent output rather than accepting it automatically.