//! Safe wrappers for AOTInductor model inference.
//!
//! This module provides type-safe Rust wrappers around the C++ FFI bindings
//! for loading and running AOTInductor-compiled PyTorch models.

use ndarray::{Array, ArrayD, IxDyn};
use std::ffi::{CStr, CString};
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use super::ffi;
use crate::error::{InductorError, Result};

/// Device specification for model inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    /// CPU device.
    Cpu,
    /// CUDA device with index (also used for ROCm via HIP compatibility).
    Cuda(usize),
}

impl Device {
    /// Create a CPU device.
    pub fn cpu() -> Self {
        Self::Cpu
    }

    /// Create a CUDA device with the given index.
    pub fn cuda(index: usize) -> Self {
        Self::Cuda(index)
    }
}

impl FromStr for Device {
    type Err = InductorError;

    /// Parse a device string like "cpu", "cuda", "cuda:0", "cuda:1".
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim().to_lowercase();
        if s == "cpu" {
            Ok(Self::Cpu)
        } else if s == "cuda" {
            Ok(Self::Cuda(0))
        } else if let Some(idx) = s.strip_prefix("cuda:") {
            let index: usize = idx
                .parse()
                .map_err(|_| InductorError::config(format!("Invalid CUDA index: {}", idx)))?;
            Ok(Self::Cuda(index))
        } else {
            Err(InductorError::config(format!("Invalid device: {}", s)))
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(idx) => write!(f, "cuda:{}", idx),
        }
    }
}

/// Tensor data extracted from inference results.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// The tensor data as a dynamic-dimensional array.
    pub data: ArrayD<f32>,
}

impl TensorData {
    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Result of model inference.
#[derive(Debug)]
pub struct InferenceResult {
    /// Output tensors from the model.
    pub outputs: Vec<TensorData>,
    /// Inference latency in milliseconds.
    pub latency_ms: f64,
    /// Peak GPU memory usage in bytes (0 for CPU).
    pub memory_bytes: usize,
}

/// Get the last error message from the C++ bridge.
fn get_last_error() -> String {
    unsafe {
        let err_ptr = ffi::aot_get_last_error();
        if err_ptr.is_null() {
            "Unknown error".to_string()
        } else {
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        }
    }
}

/// AOT-compiled model wrapper.
///
/// This struct provides a safe interface for loading and running
/// AOTInductor-compiled PyTorch models.
///
/// # Example
///
/// ```ignore
/// use inductor_rs::inference::{AotModel, Device};
/// use ndarray::Array4;
///
/// // Load model
/// let model = AotModel::load("model.pt2", Device::cpu())?;
///
/// // Create input tensor (B, C, H, W)
/// let input = Array4::<f32>::zeros((1, 1, 224, 224));
///
/// // Run inference
/// let result = model.infer(&input)?;
/// println!("Got {} outputs in {:.2}ms", result.outputs.len(), result.latency_ms);
/// ```
pub struct AotModel {
    handle: ffi::ModelHandle,
    device: Device,
}

// SAFETY: The C++ model is thread-safe for inference.
unsafe impl Send for AotModel {}
unsafe impl Sync for AotModel {}

impl AotModel {
    /// Load a model from a .pt2 package file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .pt2 model package
    /// * `device` - Device to load the model on
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to load.
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(InductorError::FileNotFound(path.to_path_buf()));
        }

        let path_str = path.to_string_lossy();
        let path_cstr = CString::new(path_str.as_ref())
            .map_err(|_| InductorError::model_load("Invalid path encoding"))?;

        let device_str = device.to_string();
        let device_cstr = CString::new(device_str)
            .map_err(|_| InductorError::model_load("Invalid device string"))?;

        let handle = unsafe { ffi::aot_model_load(path_cstr.as_ptr(), device_cstr.as_ptr()) };

        if handle.is_null() {
            Err(InductorError::model_load(format!(
                "Failed to load model: {}",
                get_last_error()
            )))
        } else {
            Ok(Self { handle, device })
        }
    }

    /// Get the device this model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run inference on an input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor as a 4D array (typically B, C, H, W)
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn infer<D: ndarray::Dimension>(&self, input: &Array<f32, D>) -> Result<InferenceResult> {
        // Create input tensor
        let shape: Vec<i64> = input.shape().iter().map(|&s| s as i64).collect();
        let device_str = self.device.to_string();
        let device_cstr = CString::new(device_str)
            .map_err(|_| InductorError::inference("Invalid device string"))?;

        // Ensure data is contiguous
        let contiguous = input.as_standard_layout();
        let data_ptr = contiguous.as_ptr();

        let input_handle = unsafe {
            ffi::aot_tensor_create(data_ptr, shape.as_ptr(), shape.len(), device_cstr.as_ptr())
        };

        if input_handle.is_null() {
            return Err(InductorError::inference(format!(
                "Failed to create input tensor: {}",
                get_last_error()
            )));
        }

        // Run inference
        let result_handle = unsafe { ffi::aot_model_infer(self.handle, input_handle) };

        // Free input tensor
        unsafe {
            ffi::aot_tensor_free(input_handle);
        }

        if result_handle.is_null() {
            return Err(InductorError::inference(format!(
                "Inference failed: {}",
                get_last_error()
            )));
        }

        // Extract outputs
        let num_outputs = unsafe { ffi::aot_result_num_outputs(result_handle) };
        let mut outputs = Vec::with_capacity(num_outputs);

        for i in 0..num_outputs {
            let tensor_handle = unsafe { ffi::aot_result_output(result_handle, i) };
            if tensor_handle.is_null() {
                unsafe { ffi::aot_result_free(result_handle) };
                return Err(InductorError::tensor(format!(
                    "Failed to get output tensor {}",
                    i
                )));
            }

            let tensor_data = unsafe { extract_tensor(tensor_handle)? };
            outputs.push(tensor_data);
        }

        // Get timing info
        let latency_ms = unsafe { ffi::aot_result_latency_ms(result_handle) };
        let memory_bytes = unsafe { ffi::aot_result_memory_bytes(result_handle) };

        // Free result
        unsafe {
            ffi::aot_result_free(result_handle);
        }

        Ok(InferenceResult {
            outputs,
            latency_ms,
            memory_bytes,
        })
    }
}

impl Drop for AotModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::aot_model_free(self.handle);
            }
        }
    }
}

/// Extract tensor data from an FFI tensor handle.
///
/// # Safety
///
/// The handle must be a valid tensor pointer from the C++ bridge.
unsafe fn extract_tensor(handle: ffi::TensorHandle) -> Result<TensorData> {
    if handle.is_null() {
        return Err(InductorError::tensor("Null tensor handle"));
    }

    let ndim = ffi::aot_tensor_ndim(handle);
    let shape_ptr = ffi::aot_tensor_shape(handle);
    let numel = ffi::aot_tensor_numel(handle);
    let data_ptr = ffi::aot_tensor_data(handle);

    if data_ptr.is_null() || shape_ptr.is_null() {
        return Err(InductorError::tensor("Null tensor data or shape"));
    }

    let shape: Vec<usize> = std::slice::from_raw_parts(shape_ptr, ndim)
        .iter()
        .map(|&s| s as usize)
        .collect();

    let data: Vec<f32> = std::slice::from_raw_parts(data_ptr, numel).to_vec();

    let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| InductorError::tensor(format!("Array shape error: {}", e)))?;

    Ok(TensorData { data: array })
}

/// Test basic tensor operations on a device.
///
/// This function tests tensor creation, basic ops (matmul, sum), and
/// empty_strided allocation. Used to diagnose HIP/CUDA issues.
///
/// Returns the test results as a string on success.
pub fn test_device(device: &Device) -> Result<String> {
    let device_str = device.to_string();
    let device_cstr =
        CString::new(device_str).map_err(|_| InductorError::config("Invalid device string"))?;

    let result = unsafe { ffi::aot_test_device(device_cstr.as_ptr()) };

    // Results are stored in the error message buffer
    let output = get_last_error();

    if result != 0 {
        Err(InductorError::inference(output))
    } else {
        Ok(output)
    }
}

/// Trait for types that can run model inference.
///
/// This trait provides a common interface for model runners,
/// allowing different implementations (e.g., batched, cached).
pub trait ModelRunner: Send + Sync {
    /// Run inference on an input tensor.
    fn infer<D: ndarray::Dimension>(&self, input: &Array<f32, D>) -> Result<InferenceResult>;

    /// Get the device this runner uses.
    fn device(&self) -> &Device;
}

impl ModelRunner for AotModel {
    fn infer<D: ndarray::Dimension>(&self, input: &Array<f32, D>) -> Result<InferenceResult> {
        self.infer(input)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
