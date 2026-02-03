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
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CMake 3.18+
- PyTorch 2.1+ (installed in Python environment)

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
    --input input.json
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
# Install training dependencies (uses CUDA PyTorch from uv.lock)
make init-training

# Build with CUDA support
make build-cuda
# or: make build FEATURES=cuda
```

#### ROCm (AMD)

```bash
# Install dependencies, then replace CUDA PyTorch with ROCm
make init-training-rocm

# Build with ROCm support
make build-rocm
# or: make build FEATURES=rocm
```

**Important:** After `make init-training-rocm`, avoid `uv sync` or `uv run` as they will revert to CUDA PyTorch from uv.lock. Use `.venv/bin/python` directly.

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
inductor-rs infer --model model.pt2 --device cpu --input input.json

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
TORCH_COMPILE_DISABLE=1 python python/scripts/export_aot.py \
    --output model-cpu.pt2 \
    --device cpu \
    --verify

# Export for CUDA (requires CUDA PyTorch and NVIDIA GPU)
TORCH_COMPILE_DISABLE=1 python python/scripts/export_aot.py \
    --output model-cuda.pt2 \
    --device cuda \
    --verify

# Export for ROCm (requires ROCm PyTorch and AMD GPU)
TORCH_COMPILE_DISABLE=1 python python/scripts/export_aot.py \
    --output model-rocm.pt2 \
    --device cuda \
    --verify
```

**Note:** AOTInductor models are device-specific. A CPU model cannot run on GPU and vice versa. The .pt2 package contains architecture-specific kernels (HSACO for ROCm, cubin for CUDA).

## GPU Backend Details

### CUDA vs ROCm Comparison

| Aspect | CUDA (NVIDIA) | ROCm (AMD) |
|--------|---------------|------------|
| Device string | `cuda:0`, `cuda:1` | `cuda:0` (HIP uses CUDA API) |
| PyTorch index | `pytorch-cuda` (cu128) | `pytorch-rocm` (rocm7.0) |
| Cargo feature | `--features cuda` | `--features rocm` |
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

Built with `--features cuda` but ROCm PyTorch installed (or vice versa).

**Fix:** Match feature to installed PyTorch:
```bash
python -c "import torch; print(torch.__version__)"
# 2.9.1+cu128 -> use --features cuda
# 2.9.1+rocm7.0 -> use --features rocm
```

## License

Apache 2.0 - See [LICENSE.md](LICENSE.md) for details.
