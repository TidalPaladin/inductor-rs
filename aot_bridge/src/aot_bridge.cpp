/**
 * @file aot_bridge.cpp
 * @brief Generic implementation of C API for AOTInductor model inference.
 *
 * This is a template implementation that supports arbitrary model outputs.
 * Customize for your specific model's output structure.
 */

#include "aot_bridge.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

// === GPU BACKEND NOTES ===
// CUDA: Uses cudaMemGetInfo(), c10::cuda::device_synchronize()
// ROCm: Uses hipMemGetInfo(), c10::hip::device_synchronize()
// Both use torch::kCUDA device type (HIP provides CUDA compatibility layer)
#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime_api.h>
#endif
#ifdef USE_HIP
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPFunctions.h>
#include <hip/hip_runtime_api.h>
#endif

/* Thread-local error message */
static thread_local std::string g_last_error;

static void set_error(const std::string &msg) { g_last_error = msg; }

/* Debug output control - enable with AOT_DEBUG=1 environment variable */
static bool is_debug_enabled() {
  static int cached = -1;
  if (cached < 0) {
    const char *env = std::getenv("AOT_DEBUG");
    cached = (env && std::string(env) == "1") ? 1 : 0;
  }
  return cached == 1;
}

#define DEBUG_PRINT(...)                                                       \
  do {                                                                         \
    if (is_debug_enabled())                                                    \
      fprintf(stderr, __VA_ARGS__);                                            \
  } while (0)

/* ============================================================================
 * Internal Structures
 * ============================================================================
 */

struct AotModel {
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  std::string device_str;
  torch::Device device;

  AotModel(std::unique_ptr<torch::inductor::AOTIModelPackageLoader> l,
           const std::string &dev_str, torch::Device dev)
      : loader(std::move(l)), device_str(dev_str), device(dev) {}
};

struct AotTensor {
  torch::Tensor tensor;
  std::vector<int64_t> shape_vec;
  mutable std::vector<float> cpu_data; // Cached CPU copy
  mutable bool cpu_data_valid = false;

  explicit AotTensor(torch::Tensor t) : tensor(std::move(t)) {
    auto sizes = tensor.sizes();
    shape_vec.assign(sizes.begin(), sizes.end());
  }
};

struct AotResult {
  std::vector<std::unique_ptr<AotTensor>> outputs;
  double latency_ms;
  size_t memory_bytes;
};

/* ============================================================================
 * Helper Functions
 * ============================================================================
 */

static torch::Device parse_device(const std::string &device_str) {
  if (device_str == "cpu") {
    return torch::kCPU;
  } else if (device_str.find("cuda") == 0) {
    if (device_str == "cuda") {
      return torch::Device(torch::kCUDA, 0);
    }
    // Parse "cuda:N"
    size_t colon = device_str.find(':');
    if (colon != std::string::npos) {
      int index = std::stoi(device_str.substr(colon + 1));
      return torch::Device(torch::kCUDA, index);
    }
    return torch::Device(torch::kCUDA, 0);
  }
  throw std::runtime_error("Invalid device: " + device_str);
}

/* ============================================================================
 * C API Implementation
 * ============================================================================
 */

extern "C" {

AotModel *aot_model_load(const char *path, const char *device_str) {
  try {
    std::string path_s(path);
    std::string dev_str(device_str);
    torch::Device device = parse_device(dev_str);

    // AOTIModelPackageLoader uses -1 for CPU, device index for CUDA/HIP
    c10::DeviceIndex device_index = device.is_cpu() ? -1 : device.index();

    // Initialize GPU device before loading model
    // This ensures the CUDA/HIP runtime is properly initialized before
    // AOTInductor tries to allocate memory or load kernels
    if (!device.is_cpu()) {
      // Create and immediately destroy a small tensor to trigger device init
      auto init_tensor =
          torch::empty({1}, torch::TensorOptions().device(device));
      // Synchronize to ensure device is fully initialized
#if defined(USE_HIP)
      c10::hip::device_synchronize();
#elif defined(USE_CUDA)
      c10::cuda::device_synchronize();
#endif
    }

    // AOTIModelPackageLoader handles all devices (CPU, CUDA, HIP)
    // It loads .pt2 packages and extracts HSACO/cubin files automatically
    auto loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        path_s,
        "model", // model name in archive
        true,    // run_single_threaded
        1,       // num_runners
        device_index);

    return new AotModel(std::move(loader), dev_str, device);
  } catch (const std::exception &e) {
    set_error(e.what());
    return nullptr;
  }
}

void aot_model_free(AotModel *model) { delete model; }

const char *aot_get_last_error(void) { return g_last_error.c_str(); }

AotTensor *aot_tensor_create(const float *data, const int64_t *shape,
                             size_t ndim, const char *device_str) {
  try {
    std::vector<int64_t> shape_vec(shape, shape + ndim);
    torch::Device device = parse_device(device_str);

    // Calculate total elements
    int64_t numel = 1;
    for (auto s : shape_vec)
      numel *= s;

    // Copy data into owned std::vector first
    std::vector<float> data_owned(data, data + numel);

    if (device.is_cpu()) {
      // CPU: use torch::tensor
      torch::Tensor tensor =
          torch::tensor(data_owned, torch::kFloat32).reshape(shape_vec);
      return new AotTensor(std::move(tensor));
    }

    // GPU: Create empty tensor and copy data
    // 1. Create empty GPU tensor
    auto gpu_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor tensor = torch::empty(shape_vec, gpu_options);

    // 2. Create CPU tensor using from_blob on owned data
    torch::Tensor cpu_tensor =
        torch::from_blob(data_owned.data(), shape_vec,
                         torch::TensorOptions().dtype(torch::kFloat32));

    // 3. Copy from CPU to GPU
    tensor.copy_(cpu_tensor);

    // Full synchronization
#if defined(USE_HIP)
    c10::hip::device_synchronize();
#elif defined(USE_CUDA)
    c10::cuda::device_synchronize();
#endif

    return new AotTensor(std::move(tensor));
  } catch (const std::exception &e) {
    set_error(e.what());
    return nullptr;
  }
}

void aot_tensor_free(AotTensor *tensor) { delete tensor; }

const float *aot_tensor_data(AotTensor *tensor) {
  if (!tensor)
    return nullptr;
  try {
    // Copy to CPU, convert to float32, and/or make contiguous if needed
    bool needs_conversion = !tensor->tensor.is_cpu() ||
                            !tensor->tensor.is_contiguous() ||
                            tensor->tensor.scalar_type() != torch::kFloat32;
    if (needs_conversion) {
      if (!tensor->cpu_data_valid) {
        torch::Tensor cpu_tensor =
            tensor->tensor.to(torch::kCPU, torch::kFloat32).contiguous();
        tensor->cpu_data.resize(cpu_tensor.numel());
        std::memcpy(tensor->cpu_data.data(), cpu_tensor.data_ptr<float>(),
                    cpu_tensor.numel() * sizeof(float));
        tensor->cpu_data_valid = true;
      }
      return tensor->cpu_data.data();
    }
    return tensor->tensor.data_ptr<float>();
  } catch (const std::exception &e) {
    set_error(e.what());
    return nullptr;
  }
}

const int64_t *aot_tensor_shape(const AotTensor *tensor) {
  if (!tensor)
    return nullptr;
  return tensor->shape_vec.data();
}

size_t aot_tensor_ndim(const AotTensor *tensor) {
  if (!tensor)
    return 0;
  return tensor->shape_vec.size();
}

size_t aot_tensor_numel(const AotTensor *tensor) {
  if (!tensor)
    return 0;
  return tensor->tensor.numel();
}

AotResult *aot_model_infer(AotModel *model, AotTensor *input) {
  if (!model || !input) {
    set_error("Null model or input");
    return nullptr;
  }

  try {
    // Get input tensor - should already be on correct device from
    // aot_tensor_create
    torch::Tensor input_tensor = input->tensor;

    // DEBUG: Print input tensor info
    DEBUG_PRINT("[DEBUG] Input tensor - mean: %.6f, std: %.6f, device: %s\n",
                input_tensor.mean().item<float>(),
                input_tensor.std().item<float>(),
                input_tensor.device().str().c_str());

    // Move to model's device if needed (should be no-op if already there)
    if (input_tensor.device() != model->device) {
      input_tensor = input_tensor.to(model->device);
    }

    // Ensure tensor is contiguous (required for AOTInductor)
    input_tensor = input_tensor.contiguous();

    // Full synchronization before inference to ensure all GPU operations
    // complete
#if defined(USE_HIP)
    if (model->device.is_cuda()) {
      c10::hip::device_synchronize();
    }
#elif defined(USE_CUDA)
    if (model->device.is_cuda()) {
      c10::cuda::device_synchronize();
    }
#endif

    // Prepare inputs
    std::vector<torch::Tensor> inputs = {input_tensor};

    // Get GPU memory usage before inference
    size_t memory_before = 0;
#if defined(USE_HIP)
    if (model->device.is_cuda()) {
      size_t free_mem, total_mem;
      hipMemGetInfo(&free_mem, &total_mem);
      memory_before = total_mem - free_mem;
    }
#elif defined(USE_CUDA)
    if (model->device.is_cuda()) {
      size_t free_mem, total_mem;
      cudaMemGetInfo(&free_mem, &total_mem);
      memory_before = total_mem - free_mem;
    }
#endif

    // Synchronize before timing
#if defined(USE_HIP)
    if (model->device.is_cuda()) {
      c10::hip::device_synchronize();
    }
#elif defined(USE_CUDA)
    if (model->device.is_cuda()) {
      c10::cuda::device_synchronize();
    }
#endif

    // Run inference with timing
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<torch::Tensor> outputs = model->loader->run(inputs);

    // Synchronize after inference
#if defined(USE_HIP)
    if (model->device.is_cuda()) {
      c10::hip::device_synchronize();
    }
#elif defined(USE_CUDA)
    if (model->device.is_cuda()) {
      c10::cuda::device_synchronize();
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // Get GPU memory usage after inference
    size_t memory_after = 0;
#if defined(USE_HIP)
    if (model->device.is_cuda()) {
      size_t free_mem, total_mem;
      hipMemGetInfo(&free_mem, &total_mem);
      memory_after = total_mem - free_mem;
    }
#elif defined(USE_CUDA)
    if (model->device.is_cuda()) {
      size_t free_mem, total_mem;
      cudaMemGetInfo(&free_mem, &total_mem);
      memory_after = total_mem - free_mem;
    }
#endif
    // Use max as peak (memory might spike during inference then get freed)
    size_t peak_memory = std::max(memory_before, memory_after);

    // Build result with all outputs
    auto result = new AotResult();
    result->outputs.reserve(outputs.size());
    for (auto &output : outputs) {
      result->outputs.push_back(std::make_unique<AotTensor>(std::move(output)));
    }
    result->latency_ms = latency_ms;
    result->memory_bytes = peak_memory;

    DEBUG_PRINT("[DEBUG] Inference complete: %zu outputs, %.2f ms\n",
                result->outputs.size(), latency_ms);

    return result;
  } catch (const std::exception &e) {
    set_error(e.what());
    return nullptr;
  }
}

void aot_result_free(AotResult *result) { delete result; }

size_t aot_result_num_outputs(const AotResult *result) {
  if (!result)
    return 0;
  return result->outputs.size();
}

AotTensor *aot_result_output(AotResult *result, size_t index) {
  if (!result || index >= result->outputs.size())
    return nullptr;
  return result->outputs[index].get();
}

double aot_result_latency_ms(const AotResult *result) {
  if (!result)
    return 0.0;
  return result->latency_ms;
}

size_t aot_result_memory_bytes(const AotResult *result) {
  if (!result)
    return 0;
  return result->memory_bytes;
}

int aot_test_device(const char *device_str) {
  try {
    torch::Device device = parse_device(device_str);
    std::string results;

    // Test each dtype
    std::vector<std::pair<torch::ScalarType, std::string>> dtypes = {
        {torch::kFloat32, "float32"},
        {torch::kFloat16, "float16"},
        {torch::kBFloat16, "bfloat16"},
    };

    results += "=== Testing device: " + std::string(device_str) + " ===\n";

    // Print device info
    if (device.is_cpu()) {
      results += "Device type: CPU\n";
    }
#if defined(USE_HIP)
    if (device.is_cuda()) {
      hipDeviceProp_t props;
      hipGetDeviceProperties(&props, device.index());
      results += "Device name: " + std::string(props.name) + "\n";
      results += "GCN arch: " + std::string(props.gcnArchName) + "\n";
      results += "Total memory: " +
                 std::to_string(props.totalGlobalMem / (1024 * 1024)) + " MB\n";
    }
#elif defined(USE_CUDA)
    if (device.is_cuda()) {
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, device.index());
      results += "Device name: " + std::string(props.name) + "\n";
      results += "Compute capability: " + std::to_string(props.major) + "." +
                 std::to_string(props.minor) + "\n";
      results += "Total memory: " +
                 std::to_string(props.totalGlobalMem / (1024 * 1024)) + " MB\n";
    }
#endif

    // Test 1: Tensor creation with randn
    results += "\n--- Tensor creation (randn) ---\n";
    for (const auto &[dtype, name] : dtypes) {
      try {
        auto x = torch::randn(
            {4, 4}, torch::TensorOptions().device(device).dtype(dtype));
        results += name + ": OK\n";
      } catch (const std::exception &e) {
        results += name + ": FAILED - " + e.what() + "\n";
      }
    }

    // Test 2: empty_strided (what AOTInductor uses)
    results += "\n--- empty_strided ---\n";
    for (const auto &[dtype, name] : dtypes) {
      try {
        auto x = torch::empty_strided(
            {2}, {1}, torch::TensorOptions().device(device).dtype(dtype));
        results += name + ": OK\n";
      } catch (const std::exception &e) {
        results += name + ": FAILED - " + e.what() + "\n";
      }
    }

    // Test 3: Matmul
    results += "\n--- Matmul ---\n";
    for (const auto &[dtype, name] : dtypes) {
      try {
        auto x = torch::randn(
            {4, 4}, torch::TensorOptions().device(device).dtype(dtype));
        auto y = torch::matmul(x, x.t());
        auto z = y.sum().item<float>();
        results += name + ": OK (sum=" + std::to_string(z) + ")\n";
      } catch (const std::exception &e) {
        results += name + ": FAILED - " + e.what() + "\n";
      }
    }

    // Test 4: Larger tensor (like model intermediate)
    results += "\n--- Large tensor ops ---\n";
    for (const auto &[dtype, name] : dtypes) {
      try {
        auto x = torch::randn(
            {4, 128, 768}, torch::TensorOptions().device(device).dtype(dtype));
        auto y = torch::randn(
            {4, 128, 768}, torch::TensorOptions().device(device).dtype(dtype));
        auto z = (x * y).sum().item<float>();
        results += name + ": OK (sum=" + std::to_string(z) + ")\n";
      } catch (const std::exception &e) {
        results += name + ": FAILED - " + e.what() + "\n";
      }
    }

    // Synchronize to catch any async errors
#if defined(USE_HIP)
    if (device.is_cuda()) {
      c10::hip::device_synchronize();
    }
#elif defined(USE_CUDA)
    if (device.is_cuda()) {
      c10::cuda::device_synchronize();
    }
#endif

    results += "\n=== All tests completed ===\n";

    // Store results in error message so Rust can retrieve it
    set_error(results);
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("Test failed: ") + e.what());
    return 1;
  }
}

} // extern "C"
