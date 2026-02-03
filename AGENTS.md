# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

### Rust

```bash
# Build the project (sets up backend venv automatically via uv)
make build

# Debug build
make build-debug

# Portable build (copies torch libraries for fully self-contained binary)
make build-portable

# Build with CUDA support
make build BACKEND=cuda

# Build with ROCm support (AMD GPUs)
make build BACKEND=rocm

# Generate test fixtures (required before running tests)
make test-fixtures

# Run Rust tests (requires test fixtures)
make test-rust

# Run a single test
cd core && cargo test test_name
```

### Code Quality

```bash
# Run all linters and checks (Rust, Python, C++)
make lint

# Format all code (Rust, Python, C++)
make fmt-all

# Individual Rust tools
make fmt           # Rust formatting (rustfmt)
make fmt-check     # Rust format check
make clippy        # Rust linting (clippy)

# Individual C++ tools
make fmt-cpp       # C++ formatting (clang-format)
make fmt-cpp-check # C++ format check
```

### Python

```bash
# Initialize CUDA environment
make init BACKEND=cuda

# Initialize ROCm environment (installs ROCm PyTorch)
make init BACKEND=rocm

# Individual Python checks
make style      # Run ruff fix and format
make quality    # Run ruff check
make types      # Run basedpyright
```

### AOT Model Export

```bash
# Export model for CPU
make export BACKEND=cpu MODEL_ARGS="--output model.pt2 --device cpu --verify"

# Export for CUDA (requires CUDA build and NVIDIA GPU)
make export BACKEND=cuda MODEL_ARGS="--output model-cuda.pt2 --device cuda --verify"

# Export for ROCm (requires ROCm build and AMD GPU)
make export BACKEND=rocm MODEL_ARGS="--output model-rocm.pt2 --device cuda --verify"
```

## Project Overview

This is a template for running AOT-compiled PyTorch Inductor models in Rust. It provides:

1. **C++ Bridge** (`aot_bridge/`) - FFI layer for AOTInductor model loading
2. **Rust Core** (`core/`) - Type-safe inference API and CLI
3. **Python Tooling** (`python/`) - Model export scripts

## Architecture

### Key Components

1. **aot_bridge/** - C++ library wrapping PyTorch's `AOTIModelPackageLoader`
   - Generic tensor handling with `std::vector<AotTensor>` outputs
   - CUDA and HIP/ROCm support with conditional compilation
   - Thread-local error handling

2. **core/build.rs** - Rust build script
   - Auto-detects PyTorch from the backend venv (`.venv`, `.venv-cuda`, `.venv-rocm`)
   - Validates feature/PyTorch compatibility
   - Manages library linking and RPATH

3. **core/src/inference/** - Rust inference module
   - `ffi.rs`: Raw FFI bindings to C++ bridge
   - `model.rs`: Safe wrappers and `ModelRunner` trait

### Key Types

**Rust:**
- `AotModel` - Loaded model handle with inference methods
- `Device` - Device specification (CPU, CUDA with index)
- `InferenceResult` - Output tensors with timing info
- `TensorData` - Dynamic-dimensional tensor wrapper

**C++:**
- `AotModel` - Wraps `AOTIModelPackageLoader`
- `AotTensor` - Tensor with CPU data caching
- `AotResult` - Vector of output tensors with metadata

## CUDA vs ROCm Technical Details

### Build System Differences

#### pyproject.toml
```toml
# CUDA is searched by default (not explicit)
[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu128"

# ROCm must be explicitly requested
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.0"
explicit = true
```

#### build.rs Feature Validation

The build script validates that the Cargo feature matches the installed PyTorch:

```rust
// Parses torch version: "2.9.1+cu128" -> CUDA, "2.9.1+rocm7.0" -> ROCm
fn validate_torch_feature_match(version: &str) {
    // Panics with helpful message if mismatch detected
}
```

#### CMakeLists.txt Linking Strategy

```cmake
# ROCm: Link directly to PyTorch's bundled .so files
if(USE_HIP)
    target_link_libraries(aot_bridge PUBLIC
        "${TORCH_LIB_PATH}/libtorch.so"
        "${TORCH_LIB_PATH}/libtorch_hip.so"
        ...
    )
else()
    # CUDA/CPU: Use cmake TORCH_LIBRARIES target
    target_link_libraries(aot_bridge PUBLIC ${TORCH_LIBRARIES})
endif()
```

### Runtime Library Loading

**CRITICAL ROCm Difference:**

ROCm libraries (rocBLAS, hipBLASLt) use `$ORIGIN` to locate data files. When torch libraries are loaded via the main binary's RPATH instead of the bridge's RUNPATH, `$ORIGIN` resolves incorrectly.

**Solution in build.rs:**
```rust
// For ROCm builds: only copy libaot_bridge.so to profile_lib_dir
// The bridge's RUNPATH points directly to torch/lib
if !cfg!(feature = "rocm") {
    // CUDA/CPU: symlink all torch libs to target/release/lib/
    for lib in torch_libs { symlink(lib, profile_lib_dir); }
}
// ROCm: libs loaded via bridge's embedded RPATH to PyTorch's lib/
```

### C++ Code Patterns

#### Conditional Compilation
```cpp
#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif
#ifdef USE_HIP
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPFunctions.h>
#endif
```

#### Device Synchronization
```cpp
// Synchronize before/after inference for accurate timing
#ifdef USE_CUDA
if (model->device.is_cuda()) {
    c10::cuda::device_synchronize();
}
#elif defined(USE_HIP)
if (model->device.is_cuda()) {  // HIP uses cuda device type
    c10::hip::device_synchronize();
}
#endif
```

#### Memory Tracking
```cpp
size_t get_gpu_memory_usage() {
    size_t free_mem, total_mem;
#ifdef USE_CUDA
    cudaMemGetInfo(&free_mem, &total_mem);
#elif defined(USE_HIP)
    hipMemGetInfo(&free_mem, &total_mem);
#endif
    return total_mem - free_mem;
}
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ROCM_PATH` | ROCm installation path | `/opt/rocm` |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit path | `/usr/local/cuda` |
| `LIBTORCH` | Manual libtorch path (bypasses Python detection) | - |
| `LIBTORCH_CXX11_ABI` | C++ ABI selection | `1` |
| `AOT_DEBUG` | Enable debug output | `0` |

## Common Issues and Solutions

### PyTorch/Feature Mismatch
```
error: PyTorch/feature mismatch detected
  Cargo feature: cuda
  PyTorch installed: ROCm (rocm7.0)
```
**Fix:** Run `make init BACKEND=cuda` for CUDA or `make init BACKEND=rocm` for ROCm.

### ROCm Library Initialization Failures
**Cause:** `$ORIGIN` resolution issues with symlinked libraries.
**Fix:** Ensure build.rs ROCm handling is correct (don't symlink torch libs for ROCm).

### "hipErrorNoBinaryForGpu"
**Cause:** Model compiled for different GPU architecture than runtime.
**Fix:** Re-export model on target hardware.

### CUDA Driver Version Issues
**Cause:** System CUDA driver older than PyTorch's requirements.
**Fix:** Update NVIDIA driver or use compatible PyTorch version.

## Customization Points

1. **Model outputs** (`aot_bridge.cpp`): Change `AotResult` structure for specific outputs
2. **Input shape** (`export_aot.py`): Modify `get_example_inputs()`
3. **Model loading** (`export_aot.py`): Implement `load_model()`
4. **Result processing** (`model.rs`): Add domain-specific output struct
5. **Input preprocessing** (`main.rs`): Add format-specific loading

## Testing Workflow

1. Generate test model: `make test-fixtures`
2. Run Rust tests: `make test-rust`
3. Run Python tests: `make test-python`
4. Full test suite: `make test`
5. Backend smoke tests: `make test-backends`

## Debugging

Enable debug output from the C++ bridge:
```bash
AOT_DEBUG=1 ./target/release/inductor-rs infer --model model.pt2 ...
```

This prints tensor statistics and timing information to stderr.
