/**
 * @file aot_bridge.h
 * @brief Generic C API for AOTInductor-compiled PyTorch model inference.
 *
 * This header defines the C interface used by Rust FFI bindings to load and
 * run inference on AOTInductor-packaged PyTorch models.
 *
 * Models are loaded from .pt2 package files (PyTorch 2 export format).
 * The .pt2 format bundles the compiled model with all necessary kernel
 * files (HSACO for ROCm, cubin for CUDA), making deployments portable.
 *
 * This is a generic template - customize the result accessors for your
 * model's specific output tensors.
 */

#ifndef AOT_BRIDGE_H
#define AOT_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque types */
typedef struct AotModel AotModel;
typedef struct AotTensor AotTensor;
typedef struct AotResult AotResult;

/* ============================================================================
 * Model Lifecycle
 * ============================================================================ */

/**
 * @brief Load an AOTInductor model from a .pt2 package.
 *
 * The .pt2 package format bundles the compiled model with all necessary
 * kernel files, making it portable across systems. The loader automatically
 * extracts and manages kernel files (HSACO for ROCm, etc.).
 *
 * @param path Path to the .pt2 model package file.
 * @param device Device string (e.g., "cpu", "cuda:0").
 * @return Pointer to loaded model, or NULL on error. Use aot_get_last_error()
 *         for error details.
 */
AotModel* aot_model_load(const char* path, const char* device);

/**
 * @brief Free a loaded model.
 *
 * @param model Pointer to model to free. Safe to call with NULL.
 */
void aot_model_free(AotModel* model);

/**
 * @brief Get the last error message.
 *
 * @return Error message string. Valid until next API call.
 */
const char* aot_get_last_error(void);

/* ============================================================================
 * Tensor Operations
 * ============================================================================ */

/**
 * @brief Create a tensor from float data.
 *
 * @param data Pointer to float data. Data is copied.
 * @param shape Pointer to shape array.
 * @param ndim Number of dimensions.
 * @param device Device string.
 * @return Pointer to created tensor, or NULL on error.
 */
AotTensor* aot_tensor_create(const float* data, const int64_t* shape, size_t ndim,
                              const char* device);

/**
 * @brief Free a tensor.
 *
 * @param tensor Pointer to tensor to free. Safe to call with NULL.
 */
void aot_tensor_free(AotTensor* tensor);

/**
 * @brief Get pointer to tensor data.
 *
 * The tensor is copied to CPU if necessary. The returned pointer is valid
 * until the tensor is freed.
 *
 * @param tensor Pointer to tensor.
 * @return Pointer to float data, or NULL on error.
 */
const float* aot_tensor_data(AotTensor* tensor);

/**
 * @brief Get tensor shape.
 *
 * @param tensor Pointer to tensor.
 * @return Pointer to shape array. Valid until tensor is freed.
 */
const int64_t* aot_tensor_shape(const AotTensor* tensor);

/**
 * @brief Get number of tensor dimensions.
 *
 * @param tensor Pointer to tensor.
 * @return Number of dimensions.
 */
size_t aot_tensor_ndim(const AotTensor* tensor);

/**
 * @brief Get total number of elements in tensor.
 *
 * @param tensor Pointer to tensor.
 * @return Number of elements.
 */
size_t aot_tensor_numel(const AotTensor* tensor);

/* ============================================================================
 * Inference
 * ============================================================================ */

/**
 * @brief Run inference on a model.
 *
 * @param model Pointer to loaded model.
 * @param input Input tensor.
 * @return Pointer to inference result, or NULL on error.
 */
AotResult* aot_model_infer(AotModel* model, AotTensor* input);

/**
 * @brief Free an inference result.
 *
 * Also frees all output tensors.
 *
 * @param result Pointer to result to free. Safe to call with NULL.
 */
void aot_result_free(AotResult* result);

/**
 * @brief Get the number of output tensors in the result.
 *
 * @param result Pointer to inference result.
 * @return Number of output tensors.
 */
size_t aot_result_num_outputs(const AotResult* result);

/**
 * @brief Get an output tensor by index.
 *
 * @param result Pointer to inference result.
 * @param index Output tensor index (0-based).
 * @return Pointer to output tensor, or NULL if index out of range.
 *         Tensor is owned by the result and freed when result is freed.
 */
AotTensor* aot_result_output(AotResult* result, size_t index);

/* ============================================================================
 * Timing Information
 * ============================================================================ */

/**
 * @brief Get inference latency in milliseconds.
 *
 * @param result Pointer to inference result.
 * @return Latency in milliseconds.
 */
double aot_result_latency_ms(const AotResult* result);

/**
 * @brief Get peak memory usage in bytes.
 *
 * Only accurate for CUDA/ROCm devices. Returns 0 for CPU.
 *
 * @param result Pointer to inference result.
 * @return Memory usage in bytes.
 */
size_t aot_result_memory_bytes(const AotResult* result);

/* ============================================================================
 * Device Testing
 * ============================================================================ */

/**
 * @brief Test basic tensor operations on a device.
 *
 * This function tests tensor creation, basic ops (matmul, sum), and
 * empty_strided allocation for each dtype (float32, float16, bfloat16).
 * Used to diagnose HIP/CUDA issues without loading a model.
 *
 * @param device Device string (e.g., "cpu", "cuda:0").
 * @return 0 on success, non-zero on failure. Use aot_get_last_error()
 *         for error details.
 */
int aot_test_device(const char* device);

#ifdef __cplusplus
}
#endif

#endif /* AOT_BRIDGE_H */
