//! Model inference module.
//!
//! This module provides FFI bindings to the C++ AOTInductor bridge
//! and safe Rust wrappers for model loading and inference.

mod ffi;
mod model;

pub use model::{test_device, AotModel, Device, InferenceResult, ModelRunner, TensorData};
