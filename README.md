# inductor-rs

A template for building Rust applications that run AOT-compiled PyTorch Inductor models.

## Overview

This template provides a complete infrastructure for running PyTorch models compiled with AOTInductor from Rust:

- **C++ Bridge**: FFI layer using `torch::inductor::AOTIModelPackageLoader` for .pt2 packages
- **Rust Core**: Type-safe inference with CLI, configuration, and ModelRunner trait
- **Python Tooling**: pyproject.toml, Makefile, and AOT export scripts
- **GPU Support**: Both CUDA (NVIDIA) and ROCm (AMD) backends

## Prerequisites

- Rust 1.70+ (2021 edition)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CMake 3.18+
- PyTorch 2.1+ (installed in Python environment)

## Python Dependencies

Dependencies are split into groups in `pyproject.toml` so training, export, and dev tooling stay isolated:

- Base `project.dependencies`: minimal shared deps for AOT export/inference (keep this lean).
- `training`: training-only libraries.
- `export`: extra deps needed only for AOT export scripts.
- `dev`: linting/testing/tooling.

Examples:

```bash
# Minimal AOT export environment (base + export group)
uv sync --group export

# Training + dev environment
uv sync --group training --group dev

# Export + dev tooling
uv sync --group export --group dev
```

`make init` uses the training + dev groups by default. If you want a minimal export-only environment, run `uv sync --group export` directly (or adjust the Makefile for your workflow).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/inductor-rs.git
cd inductor-rs

# Build (automatically sets up Python venv with PyTorch)
make

# Generate a test model
make test-fixtures

# Run inference
./target/release/inductor-rs infer \
    --model tests/fixtures/dummy_model.pt2 \
    --device cpu \
    --input tests/fixtures/input.json
```

## Building

### Basic Build

```bash
# Release build (default)
make build

# Debug build
make build-debug

# Portable build (copies all libraries for distribution)
make build-portable
```

### GPU Backend Selection

#### CUDA (NVIDIA)

```bash
# Initialize CUDA environment
make init BACKEND=cuda

# Build with CUDA support
make build BACKEND=cuda
```

#### ROCm (AMD)

```bash
# Initialize ROCm environment (installs ROCm PyTorch)
make init BACKEND=rocm

# Build with ROCm support
make build BACKEND=rocm
```

**Note:** CUDA uses `.venv-cuda` and ROCm uses `.venv-rocm`. Use the backend-specific `make` targets (or the venv's `python`) to avoid mixing environments.

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LIBTORCH` | Manual libtorch path (bypasses Python detection) | Auto-detect |
| `LIBTORCH_CXX11_ABI` | C++ ABI selection | `1` |
| `ROCM_PATH` | ROCm installation path | `/opt/rocm` |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit path | `/usr/local/cuda` |
| `AOT_DEBUG` | Enable debug output | `0` |
| `AOT_PORTABLE` | Copy libraries instead of symlinking | `0` |

## Usage

### CLI Commands

```bash
# Run inference
inductor-rs infer --model model.pt2 --device cpu --input tests/fixtures/input.json

# Show model info
inductor-rs info --model model.pt2

# Test device capabilities
inductor-rs test-device --device cuda:0
```

### Input Format

Input data is provided as JSON:

```json
{
    "data": [0.1, 0.2, 0.3, ...],
    "shape": [1, 1, 224, 224]
}
```

The length of `data` must match the product of the `shape` dimensions.

### Configuration (Template)

`--config` and `config/example.yaml` are illustrative only and not applied by the CLI.
Downstream projects should either wire configuration into the CLI or remove it.

### Rust API

```rust
use inductor_rs::{AotModel, Device};
use ndarray::Array4;

// Load model
let model = AotModel::load("model.pt2", Device::cpu())?;

// Create input tensor (B, C, H, W)
let input = Array4::<f32>::zeros((1, 1, 224, 224));

// Run inference
let result = model.infer(&input)?;
println!("Got {} outputs in {:.2}ms", result.outputs.len(), result.latency_ms);
```

## AOT Model Export

Models must be exported to AOTInductor format before use:

```bash
# Export for CPU
make export BACKEND=cpu MODEL_ARGS="--output model-cpu.pt2 --device cpu --verify"

# Export for CUDA (requires CUDA PyTorch and NVIDIA GPU)
make export BACKEND=cuda MODEL_ARGS="--output model-cuda.pt2 --device cuda --verify"

# Export for ROCm (requires ROCm PyTorch and AMD GPU)
make export BACKEND=rocm MODEL_ARGS="--output model-rocm.pt2 --device cuda --verify"
```

**Note:** AOTInductor models are device-specific. A CPU model cannot run on GPU and vice versa. The .pt2 package contains architecture-specific kernels (HSACO for ROCm, cubin for CUDA).

### Export Requirements and Gotchas

- The exported model must return a plain `tuple[Tensor, ...]`. Wrap your model if it returns a dict, dataclass, or a single tensor.
- Avoid data-dependent control flow in `forward` (use `torch.where`, not `if`).
- Cast outputs to a consistent dtype (typically `float32`).
- Store sizes as Python `int`, not tensors, to avoid guard errors.
- If you have `@torch.compile` decorators, set `TORCH_COMPILE_DISABLE=1` during export to avoid FakeTensorMode conflicts.

### Verify the Export

You can sanity-check a `.pt2` package with the AOTInductor loader:

```python
from torch._inductor import aoti_load_package

runner = aoti_load_package("model.pt2", device_index=-1)  # -1 for CPU
outputs = runner(example_input)
for out in outputs:
    assert not out.isnan().any()
    assert not out.isinf().any()
```

### Portable Distribution

For fully portable builds, use the `AOT_PORTABLE=1` environment variable. This copies runtime libraries into `target/portable/.../lib/` instead of creating symlinks:

```bash
AOT_PORTABLE=1 make build-portable
```

The binary embeds two RPATH entries:
- Absolute path for the build machine (works immediately after build).
- `$ORIGIN/lib` for deployed bundles.

If you need a fully relocatable bundle, patch RPATH to use only `$ORIGIN/lib` and set RPATH on copied libraries (requires `patchelf`):

```bash
patchelf --set-rpath '$ORIGIN/lib' target/portable/release/inductor-rs
for lib in target/portable/release/lib/*.so*; do
    if [ -f "$lib" ] && ! [ -L "$lib" ]; then
        patchelf --set-rpath '$ORIGIN' "$lib" 2>/dev/null || true
    fi
done
```

## GPU Backend Details

### CUDA vs ROCm Comparison

| Aspect | CUDA (NVIDIA) | ROCm (AMD) |
|--------|---------------|------------|
| Device string | `cuda:0`, `cuda:1` | `cuda:0` (HIP uses CUDA API) |
| PyTorch index | `pytorch-cuda` (cu128) | `pytorch-rocm` (rocm7.0) |
| Make backend | `BACKEND=cuda` | `BACKEND=rocm` |
| C++ define | `USE_CUDA` | `USE_HIP`, `__HIP_PLATFORM_AMD__` |

### Library Loading

**CUDA builds:** Torch libraries are symlinked to `target/release/lib/` for portable deployment.

**ROCm builds:** Torch libraries are NOT symlinked. ROCm libraries use `$ORIGIN` to locate data files, which fails when loaded via symlinks. The bridge's RPATH points directly to `torch/lib`.

### Common GPU Architectures

| Architecture | GPUs |
|--------------|------|
| `gfx1100` | RX 7900 series (RDNA 3) |
| `gfx1030` | RX 6800/6900 series (RDNA 2) |
| `gfx90a` | MI200 series |

## Project Structure

```
inductor-rs/
├── aot_bridge/           # C++ FFI bridge
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── aot_bridge.h
│   └── src/
│       └── aot_bridge.cpp
├── core/                 # Rust crate
│   ├── Cargo.toml
│   ├── build.rs
│   └── src/
│       ├── lib.rs
│       ├── main.rs
│       ├── cli.rs
│       ├── config.rs
│       ├── error.rs
│       └── inference/
│           ├── mod.rs
│           ├── ffi.rs
│           └── model.rs
├── python/               # Python tooling
│   └── scripts/
│       ├── export_aot.py
│       └── generate_test_model.py
├── config/
│   └── example.yaml
├── pyproject.toml        # Python dependencies (uv)
├── Cargo.toml            # Workspace manifest
├── Makefile
├── README.md
└── CLAUDE.md
```

## Customization

### 1. Model Outputs

Edit `aot_bridge/src/aot_bridge.cpp` to handle your model's specific output tensors. The template uses a generic `std::vector<AotTensor>` for arbitrary outputs.

### 2. Input Shape

Edit `python/scripts/export_aot.py`:
- Modify `get_example_inputs()` to match your model's expected input
- Implement `load_model()` to load your actual model

### 3. Model Loading

Replace the dummy model in `load_model()` with your actual model loading code:

```python
def load_model() -> nn.Module:
    model = YourModel()
    model.load_state_dict(torch.load("checkpoint.pt"))
    return model
```

### 4. Result Processing

Edit `core/src/inference/model.rs` to add domain-specific output structures.

## Optional Model Bundling Pattern

If you want to ship models alongside the binary, a common pattern is:
- Bundle a `models/` directory next to the executable.
- Add a small lookup layer to select a preset (for example, `models/default/model.pt2`).
- Allow overriding the models path with an environment variable (for example, `MODELS_DIR`).

This repository does not implement model discovery yet, but the structure above keeps deployment simple and works well with portable builds.

## Testing

```bash
# Generate test fixtures
make test-fixtures

# Run Rust tests
make test-rust

# Run Python tests
make test-python

# Run all tests
make test

# Run CUDA + ROCm backend smoke tests (skips if GPU unavailable)
make test-backends
```

If you have multiple GPUs, you can force a specific device for smoke tests:

```bash
# Run ROCm smoke test on device 1
BACKEND=rocm BACKEND_DEVICE_INDEX=1 make test-backend
```

## Code Quality

```bash
# Format all code
make fmt-all

# Run all linters
make lint
```

## Troubleshooting

### ROCm: "hipErrorNoBinaryForGpu"

Model compiled for different GPU architecture than runtime.

**Fix:** Re-export model on target hardware.

### ROCm: Library initialization failures

`$ORIGIN` resolution issues with symlinked libraries.

**Fix:** Ensure ROCm builds don't symlink torch libs (handled by build.rs).

### CUDA: "CUDA driver version insufficient"

System CUDA driver older than PyTorch's requirements.

**Fix:** Update NVIDIA driver or use compatible PyTorch version.

### Feature mismatch errors

Built with `BACKEND=cuda` but ROCm PyTorch installed (or vice versa).

**Fix:** Match feature to installed PyTorch:
```bash
python -c "import torch; print(torch.__version__)"
# 2.9.1+cu128 -> use BACKEND=cuda
# 2.9.1+rocm7.0 -> use BACKEND=rocm
```

### Export failures with FakeTensorMode

If export fails with FakeTensorMode errors, disable `@torch.compile` during export:

```bash
TORCH_COMPILE_DISABLE=1 make export BACKEND=cpu MODEL_ARGS="--output model.pt2 --device cpu"
```

## License

Apache 2.0 - See [LICENSE.md](LICENSE.md) for details.
