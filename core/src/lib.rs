//! inductor-rs: Template for running AOT-compiled PyTorch Inductor models in Rust.
//!
//! This crate provides a Rust interface for loading and running PyTorch models
//! that have been compiled with AOTInductor. It uses a C++ bridge to interface
//! with PyTorch's AOTIModelPackageLoader.
//!
//! # Features
//!
//! - **cuda**: Enable CUDA GPU support (requires CUDA toolkit and PyTorch with CUDA)
//! - **rocm**: Enable ROCm/HIP GPU support (requires ROCm and PyTorch with ROCm)
//!
//! # Example
//!
//! ```ignore
//! use inductor_rs::inference::{AotModel, Device};
//! use ndarray::Array4;
//!
//! // Load model
//! let model = AotModel::load("model.pt2", Device::cpu())?;
//!
//! // Create input tensor (B, C, H, W)
//! let input = Array4::<f32>::zeros((1, 1, 224, 224));
//!
//! // Run inference
//! let result = model.infer(&input)?;
//! println!("Got {} outputs in {:.2}ms", result.outputs.len(), result.latency_ms);
//! ```
//!
//! # Building
//!
//! This crate requires PyTorch to be installed in the Python environment.
//! The build script will automatically detect PyTorch from `.venv/bin/python`.
//!
//! ```bash
//! # Create venv and install PyTorch
//! python -m venv .venv
//! .venv/bin/pip install torch
//!
//! # Build
//! cargo build --release
//!
//! # With CUDA support
//! cargo build --release --features cuda
//!
//! # With ROCm support
//! cargo build --release --features rocm
//! ```

pub mod cli;
pub mod config;
pub mod error;
pub mod inference;

// Re-export commonly used types
pub use error::{InductorError, Result};
pub use inference::{AotModel, Device, InferenceResult, TensorData};
