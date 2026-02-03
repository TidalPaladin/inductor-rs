//! FFI declarations for the C++ AOTInductor bridge.
//!
//! This module contains the raw FFI bindings. Use the safe wrappers
//! in the `model` module instead of calling these directly.

use std::ffi::c_void;
use std::os::raw::c_char;

/// Opaque handle to the C++ model.
pub type ModelHandle = *mut c_void;

/// Opaque handle to a C++ tensor.
pub type TensorHandle = *mut c_void;

/// Opaque handle to inference result.
pub type ResultHandle = *mut c_void;

extern "C" {
    // Model lifecycle
    pub fn aot_model_load(path: *const c_char, device: *const c_char) -> ModelHandle;
    pub fn aot_model_free(model: ModelHandle);
    pub fn aot_get_last_error() -> *const c_char;

    // Tensor operations
    pub fn aot_tensor_create(
        data: *const f32,
        shape: *const i64,
        ndim: usize,
        device: *const c_char,
    ) -> TensorHandle;
    pub fn aot_tensor_free(tensor: TensorHandle);
    pub fn aot_tensor_data(tensor: TensorHandle) -> *const f32;
    pub fn aot_tensor_shape(tensor: TensorHandle) -> *const i64;
    pub fn aot_tensor_ndim(tensor: TensorHandle) -> usize;
    pub fn aot_tensor_numel(tensor: TensorHandle) -> usize;

    // Inference
    pub fn aot_model_infer(model: ModelHandle, input: TensorHandle) -> ResultHandle;
    pub fn aot_result_free(result: ResultHandle);
    pub fn aot_result_num_outputs(result: ResultHandle) -> usize;
    pub fn aot_result_output(result: ResultHandle, index: usize) -> TensorHandle;

    // Timing
    pub fn aot_result_latency_ms(result: ResultHandle) -> f64;
    pub fn aot_result_memory_bytes(result: ResultHandle) -> usize;

    // Device testing
    pub fn aot_test_device(device: *const c_char) -> i32;
}
