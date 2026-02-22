//! Build script for inductor-rs.
//!
//! This script builds the C++ AOTInductor bridge library and links it to the Rust crate.
//! PyTorch/libtorch is required for building.
//!
//! # GPU Backend Support
//!
//! ## CUDA (NVIDIA)
//! - Cargo feature: `cuda` (set via `BACKEND=cuda` in the Makefile)
//! - PyTorch index: pytorch-cuda (cu128)
//! - Env: CUDA_HOME, CUDA_PATH
//! - Libs symlinked to target/release/lib/
//!
//! ## ROCm (AMD)
//! - Cargo feature: `rocm` (set via `BACKEND=rocm` in the Makefile)
//! - PyTorch index: pytorch-rocm (rocm7.0)
//! - Env: ROCM_PATH
//! - CRITICAL: Libs NOT symlinked (ROCm $ORIGIN resolution issues)
//! - Bridge RPATH points to torch/lib directly
//!
//! # Environment Variables
//!
//! - `LIBTORCH`: Path to libtorch installation (optional, auto-detected from Python if not set)
//! - `LIBTORCH_CXX11_ABI`: Set to "1" to use the CXX11 ABI (default: "1")
//! - `AOT_BRIDGE_SKIP_BUILD`: Set to "1" to skip building (for development)
//! - `ROCM_PATH`: Path to ROCm installation (optional, for ROCm builds)

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// PyTorch device variant detected from version string.
#[derive(Debug, Clone, PartialEq)]
enum TorchDevice {
    Cuda(String), // e.g., "cu128"
    Rocm(String), // e.g., "rocm7.0"
    Cpu,
}

/// Parse the device variant from a torch version string.
/// Examples: "2.9.1+cu128" -> Cuda("cu128"), "2.9.1+rocm7.0" -> Rocm("rocm7.0")
fn parse_torch_device(version: &str) -> TorchDevice {
    if let Some(suffix) = version.split('+').nth(1) {
        if suffix.starts_with("cu") {
            TorchDevice::Cuda(suffix.to_string())
        } else if suffix.starts_with("rocm") {
            TorchDevice::Rocm(suffix.to_string())
        } else {
            TorchDevice::Cpu
        }
    } else {
        TorchDevice::Cpu
    }
}

/// Validate that the cargo feature matches the installed PyTorch variant.
/// Panics with a helpful error message if there's a mismatch.
fn validate_torch_feature_match(version: &str) {
    let device = parse_torch_device(version);
    let feature_cuda = cfg!(feature = "cuda");
    let feature_rocm = cfg!(feature = "rocm");

    // No GPU feature enabled - compatible with any PyTorch variant
    if !feature_cuda && !feature_rocm {
        return;
    }

    let (required, installed, fix_cmd) = match (&device, feature_cuda, feature_rocm) {
        // CUDA feature but non-CUDA PyTorch
        (TorchDevice::Rocm(v), true, false) => {
            ("CUDA", format!("ROCm ({})", v), "make init BACKEND=cuda")
        }
        (TorchDevice::Cpu, true, false) => ("CUDA", "CPU".to_string(), "make init BACKEND=cuda"),
        // ROCm feature but non-ROCm PyTorch
        (TorchDevice::Cuda(v), false, true) => {
            ("ROCm", format!("CUDA ({})", v), "make init BACKEND=rocm")
        }
        (TorchDevice::Cpu, false, true) => ("ROCm", "CPU".to_string(), "make init BACKEND=rocm"),
        // Match - no error
        _ => return,
    };

    let feature_name = if feature_cuda { "cuda" } else { "rocm" };

    panic!(
        "\n\
        \n\
        error: PyTorch/feature mismatch detected\n\
        \n\
          Cargo feature: {feature_name}\n\
          PyTorch installed: {installed}\n\
          PyTorch version: {version}\n\
        \n\
        The installed PyTorch does not have {required} support.\n\
        \n\
        To fix this, either:\n\
        \n\
          1. Install PyTorch with {required} support:\n\
             {fix_cmd}\n\
        \n\
          2. Or build without the {feature_name} feature:\n\
             cargo build --release\n\
        \n\
        See CLAUDE.md for more details.\n"
    );
}

/// Detect libtorch from the current Python environment.
///
/// Tries multiple Python executables in order:
/// 1. Backend venvs (`.venv-cuda`, `.venv-rocm`, `.venv`) in repo root
/// 2. Backend venvs in `core/` (if present)
/// 3. `python3` - system Python 3
/// 4. `python` - fallback
///
/// Returns (torch_path, python_executable) on success.
fn detect_libtorch_from_python() -> Option<(PathBuf, PathBuf)> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Try these Python executables in order
    let mut python_candidates = Vec::new();

    // Prefer backend-specific venvs, then fall back to the default .venv
    let venv_names: &[&str] = if cfg!(feature = "cuda") {
        &[".venv-cuda", ".venv"]
    } else if cfg!(feature = "rocm") {
        &[".venv-rocm", ".venv"]
    } else {
        &[".venv", ".venv-cuda", ".venv-rocm"]
    };

    for name in venv_names {
        python_candidates.push(manifest_dir.join(format!("../{}/bin/python", name)));
        python_candidates.push(manifest_dir.join(format!("{}/bin/python", name)));
    }

    python_candidates.push(PathBuf::from("python3"));
    python_candidates.push(PathBuf::from("python"));

    for python in &python_candidates {
        let output = Command::new(python)
            .args(["-c", "import torch; print(torch.__path__[0])"])
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                if let Ok(stdout) = String::from_utf8(output.stdout) {
                    let path = stdout.trim().to_string();
                    let torch_path = PathBuf::from(&path);
                    // Verify it has the cmake config we need
                    if torch_path
                        .join("share/cmake/Torch/TorchConfig.cmake")
                        .exists()
                    {
                        return Some((torch_path, python.clone()));
                    }
                }
            }
        }
    }
    None
}

/// Query PyTorch version from Python and emit as cargo env var.
/// Returns the version string for further processing (e.g., feature validation).
fn emit_libtorch_version(python: &PathBuf) -> Option<String> {
    let output = Command::new(python)
        .args(["-c", "import torch; print(torch.__version__)"])
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                let version = stdout.trim().to_string();
                if !version.is_empty() {
                    println!("cargo:rustc-env=LIBTORCH_VERSION={}", version);
                    return Some(version);
                }
            }
        }
    }
    None
}

fn main() {
    build_aot_bridge();
}

fn build_aot_bridge() {
    // Check if we should skip the build
    if env::var("AOT_BRIDGE_SKIP_BUILD")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        println!("cargo:warning=Skipping aot-bridge build (AOT_BRIDGE_SKIP_BUILD=1)");
        return;
    }

    // Get libtorch path: try LIBTORCH env var first, then auto-detect from Python
    let (libtorch, python) = if let Ok(path) = env::var("LIBTORCH") {
        // When LIBTORCH is set manually, try to find Python for version info
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let mut python_candidates = Vec::new();
        let venv_names = [".venv", ".venv-cuda", ".venv-rocm"];
        for name in venv_names {
            python_candidates.push(manifest_dir.join(format!("../{}/bin/python", name)));
            python_candidates.push(manifest_dir.join(format!("{}/bin/python", name)));
        }
        python_candidates.push(PathBuf::from("python3"));
        python_candidates.push(PathBuf::from("python"));

        let python = python_candidates
            .into_iter()
            .find(|p| Command::new(p).arg("--version").output().is_ok());
        (PathBuf::from(path), python)
    } else if let Some((path, python)) = detect_libtorch_from_python() {
        let msg = format!("Auto-detected PyTorch from Python: {}", path.display());
        let warn = env::var("AOT_WARN_LTORCH")
            .map(|v| v == "1")
            .unwrap_or(false);
        if warn {
            println!("cargo:warning={}", msg);
        } else {
            eprintln!("info: {}", msg);
        }
        (path, Some(python))
    } else {
        panic!(
            "Could not find PyTorch installation.\n\
             Checked: .venv/.venv-cuda/.venv-rocm (repo and core), python3, python\n\
             \n\
             To fix, either:\n\
             1. Create a venv with PyTorch: make init BACKEND=cpu|cuda|rocm\n\
             2. Set LIBTORCH environment variable to your PyTorch installation"
        );
    };

    // Emit libtorch version for --version output and validate feature compatibility
    if let Some(ref python) = python {
        if let Some(version) = emit_libtorch_version(python) {
            validate_torch_feature_match(&version);
        }
    }

    if !libtorch.exists() {
        panic!(
            "Libtorch path does not exist: {}\n\
             The LIBTORCH environment variable points to a non-existent path.",
            libtorch.display()
        );
    }

    // Rerun if bridge sources change
    println!("cargo:rerun-if-changed=../aot_bridge/src/aot_bridge.cpp");
    println!("cargo:rerun-if-changed=../aot_bridge/include/aot_bridge.h");
    println!("cargo:rerun-if-changed=../aot_bridge/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=AOT_BRIDGE_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=AOT_PORTABLE");

    // Get the bridge directory
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bridge_dir = manifest_dir.join("../aot_bridge");
    let lib_dir = libtorch.join("lib");

    // Build with CMake
    let mut cmake_config = cmake::Config::new(&bridge_dir);

    // Set libtorch path and force Torch discovery to this prefix
    cmake_config.define("CMAKE_PREFIX_PATH", &libtorch);
    cmake_config.define(
        "Torch_DIR",
        libtorch.join("share/cmake/Torch").to_str().unwrap(),
    );

    // Pass torch lib path for RPATH in the bridge library
    cmake_config.define("TORCH_LIB_PATH", lib_dir.to_str().unwrap());

    // Set build type
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let build_type = if profile == "release" {
        "Release"
    } else {
        "Debug"
    };
    cmake_config.define("CMAKE_BUILD_TYPE", build_type);

    // Handle CXX11 ABI - default to 1 for modern PyTorch
    let cxx11_abi = env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_string());
    cmake_config.define(
        "CMAKE_CXX_FLAGS",
        format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi),
    );

    // Ensure backend switches reset cached CMake options.
    cmake_config.define(
        "USE_CUDA",
        if cfg!(feature = "cuda") { "ON" } else { "OFF" },
    );
    cmake_config.define("USE_HIP", if cfg!(feature = "rocm") { "ON" } else { "OFF" });

    // Check for ROCm feature
    if cfg!(feature = "rocm") {
        // Pass ROCM_PATH to CMake if set
        if let Ok(rocm_path) = env::var("ROCM_PATH") {
            cmake_config.define("ROCM_PATH", rocm_path);
        }
    }

    // Avoid mixing stale CUDA/HIP settings when switching backends.
    // By default, the cmake crate reuses its cache unless this is disabled.
    cmake_config.always_configure(true);

    // Check if libtorch has CUDA support by looking for CUDA cmake files
    let libtorch_has_cuda = libtorch
        .join("share/cmake/Caffe2/public/cuda.cmake")
        .exists();
    // Check if libtorch has HIP/ROCm support - use actual library presence
    let libtorch_has_hip = libtorch.join("lib/libtorch_hip.so").exists() || cfg!(feature = "rocm");

    // Find CUDA toolkit from CUDA_HOME, CUDA_PATH, or common paths
    let cuda_root = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            ["/opt/cuda", "/usr/local/cuda"]
                .iter()
                .map(PathBuf::from)
                .find(|p| p.exists())
        });

    // Set CUDA architectures if libtorch has CUDA support (not ROCm)
    if libtorch_has_cuda && !libtorch_has_hip {
        if let Some(ref cuda_path) = cuda_root {
            // Tell CMake's FindCUDA where the toolkit is
            cmake_config.define("CUDA_TOOLKIT_ROOT_DIR", cuda_path);
            cmake_config.define("CMAKE_CUDA_ARCHITECTURES", "native");

            // Use GCC 13 if available (CUDA 12.x doesn't support GCC 14+)
            if std::path::Path::new("/usr/bin/gcc-13").exists() {
                cmake_config.define("CMAKE_C_COMPILER", "/usr/bin/gcc-13");
                cmake_config.define("CMAKE_CXX_COMPILER", "/usr/bin/g++-13");
                cmake_config.define("CMAKE_CUDA_HOST_COMPILER", "/usr/bin/g++-13");
            }
        }
    }

    // Build
    let dst = cmake_config.build();

    // Link the bridge library ONLY - it will load torch libraries via its own rpath
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=aot_bridge");

    // Still need search path for rpath resolution
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Use RPATH (not RUNPATH) for reliable library discovery
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");

    // Add absolute path to libtorch for the build machine
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // Add relative rpath for portable deployment
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");

    // === Setup runtime library directory ===
    // Copy bridge library and symlink/copy torch libs to target/{profile}/lib/
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile_dir = out_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("Could not determine profile directory from OUT_DIR");
    let profile_lib_dir = profile_dir.join("lib");

    // AOT_PORTABLE=1 copies libraries instead of symlinking for fully portable builds
    let portable = env::var("AOT_PORTABLE").map(|v| v == "1").unwrap_or(false);

    if portable {
        println!("cargo:warning=AOT_PORTABLE=1: Copying libraries for portable build");
    }

    if let Err(e) = std::fs::create_dir_all(&profile_lib_dir) {
        println!("cargo:warning=Failed to create lib dir: {}", e);
    } else {
        // Copy the bridge library (not symlink - it's our build artifact)
        let bridge_src = dst.join("lib/libaot_bridge.so");
        let bridge_dst = profile_lib_dir.join("libaot_bridge.so");
        if let Err(e) = std::fs::copy(&bridge_src, &bridge_dst) {
            println!("cargo:warning=Failed to copy bridge library: {}", e);
        }

        // For ROCm builds, ensure no stale CUDA libs remain in the profile lib dir.
        // This avoids loading mixed CUDA/ROCm libraries via $ORIGIN.
        if cfg!(feature = "rocm") {
            if let Ok(entries) = std::fs::read_dir(&profile_lib_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name == "libaot_bridge.so" {
                        continue;
                    }
                    let path = entry.path();
                    let _ = if path.is_dir() {
                        std::fs::remove_dir_all(&path)
                    } else {
                        std::fs::remove_file(&path)
                    };
                }
            }
        }

        // Helper to copy or symlink a library
        let link_or_copy = |src: &std::path::Path, dst: &std::path::Path| {
            if portable {
                // Follow symlinks when copying (src might be a symlink itself)
                if let Ok(real_src) = std::fs::canonicalize(src) {
                    let _ = std::fs::copy(&real_src, dst);
                } else {
                    let _ = std::fs::copy(src, dst);
                }
            } else {
                let _ = std::os::unix::fs::symlink(src, dst);
            }
        };

        // IMPORTANT: For ROCm builds, we do NOT symlink torch libraries here.
        // ROCm inference fails when torch libs are loaded via the main binary's RPATH
        // instead of the bridge's RUNPATH. The root cause is that rocBLAS and other
        // ROCm libraries use $ORIGIN to locate data files, and when loaded via symlinks,
        // $ORIGIN resolves to the wrong directory causing initialization issues.
        //
        // The fix: only put libaot_bridge.so in profile_lib_dir for ROCm builds.
        // The bridge's RUNPATH points directly to torch/lib, which works correctly.
        //
        // For CUDA and CPU builds, symlinking works fine.
        if !cfg!(feature = "rocm") {
            if let Ok(entries) = std::fs::read_dir(&lib_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.contains(".so") {
                        let dst_path = profile_lib_dir.join(&name);
                        if !dst_path.exists() {
                            link_or_copy(&entry.path(), &dst_path);
                        }
                    }
                }
            }

            // Also symlink/copy data directories that ROCm libraries need
            let data_dirs = ["rocblas", "hipblaslt", "hipsparselt", "aotriton.images"];
            for dir_name in data_dirs {
                let src_dir = lib_dir.join(dir_name);
                let dst_dir = profile_lib_dir.join(dir_name);
                if src_dir.exists() && !dst_dir.exists() {
                    if portable {
                        fn copy_dir_recursive(
                            src: &std::path::Path,
                            dst: &std::path::Path,
                        ) -> std::io::Result<()> {
                            std::fs::create_dir_all(dst)?;
                            for entry in std::fs::read_dir(src)? {
                                let entry = entry?;
                                let src_path = entry.path();
                                let dst_path = dst.join(entry.file_name());
                                if src_path.is_dir() {
                                    copy_dir_recursive(&src_path, &dst_path)?;
                                } else {
                                    std::fs::copy(&src_path, &dst_path)?;
                                }
                            }
                            Ok(())
                        }
                        let _ = copy_dir_recursive(&src_dir, &dst_dir);
                    } else {
                        let _ = std::os::unix::fs::symlink(&src_dir, &dst_dir);
                    }
                }
            }
        }

        // Symlink/copy NVIDIA libraries from pip packages (for CUDA support, not ROCm)
        if !cfg!(feature = "rocm") {
            let nvidia_base = lib_dir
                .parent()
                .and_then(|p| p.parent())
                .map(|p| p.join("nvidia"));
            if let Some(nvidia_dir) = nvidia_base {
                if nvidia_dir.exists() {
                    if let Ok(nvidia_packages) = std::fs::read_dir(&nvidia_dir) {
                        for package in nvidia_packages.flatten() {
                            let package_lib = package.path().join("lib");
                            if package_lib.exists() {
                                if let Ok(libs) = std::fs::read_dir(&package_lib) {
                                    for lib in libs.flatten() {
                                        let name = lib.file_name().to_string_lossy().to_string();
                                        if name.contains(".so") {
                                            let dst_path = profile_lib_dir.join(&name);
                                            if !dst_path.exists() {
                                                link_or_copy(&lib.path(), &dst_path);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
