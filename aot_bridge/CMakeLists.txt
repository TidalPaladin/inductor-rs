cmake_minimum_required(VERSION 3.18)
project(aot_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find libtorch
# Set LIBTORCH environment variable or CMAKE_PREFIX_PATH to your libtorch installation
if(DEFINED ENV{LIBTORCH})
    list(APPEND CMAKE_PREFIX_PATH $ENV{LIBTORCH})
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Ensure we use the same C++ ABI as libtorch (new CXX11 ABI)
# This is required to avoid symbol mismatch errors with std::string
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# === GPU BACKEND NOTES ===
# CUDA: Uses cudaMemGetInfo(), c10::cuda::device_synchronize()
# ROCm: Uses hipMemGetInfo(), c10::hip::device_synchronize()
# Both use torch::kCUDA device type (HIP provides CUDA compatibility layer)

# Check for CUDA or HIP/ROCm support
# USE_HIP must be explicitly set (from build.rs based on cargo feature)
# Don't auto-detect from ROCM_PATH since it may be installed alongside CUDA
if(USE_HIP)
    # ROCm/HIP build - PyTorch was built with HIP support
    message(STATUS "HIP/ROCm support enabled (explicit)")
    add_compile_definitions(USE_HIP)
    # Required for HIP headers to use AMD platform definitions
    add_compile_definitions(__HIP_PLATFORM_AMD__)
    # Add ROCm include path for HIP headers
    if(DEFINED ROCM_PATH)
        set(ROCM_INCLUDE_DIR "${ROCM_PATH}/include")
    elseif(DEFINED ENV{ROCM_PATH})
        set(ROCM_INCLUDE_DIR "$ENV{ROCM_PATH}/include")
    else()
        set(ROCM_INCLUDE_DIR "/opt/rocm/include")
    endif()
    message(STATUS "ROCm include dir: ${ROCM_INCLUDE_DIR}")
elseif(USE_CUDA OR TORCH_CUDA_LIBRARIES)
    message(STATUS "CUDA support enabled")
    add_compile_definitions(USE_CUDA)
else()
    message(STATUS "GPU support disabled (CPU only)")
endif()

# Build the bridge library
add_library(aot_bridge SHARED
    src/aot_bridge.cpp
)

target_include_directories(aot_bridge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
        ${ROCM_INCLUDE_DIR}
)

# For ROCm builds, we link directly to PyTorch's bundled .so files instead of using
# cmake targets. This is necessary because cmake's find_package(Torch) populates
# torch targets with INTERFACE_LINK_LIBRARIES pointing to system ROCm libs (which may
# be a different version than PyTorch's bundled ones).
#
# PyTorch ROCm wheels are self-contained - they bundle all required ROCm libraries.
if(USE_HIP)
    # TORCH_LIB_PATH is passed from build.rs and points to PyTorch's lib directory
    # Link directly to PyTorch's bundled libraries by path
    # This avoids cmake targets pulling in system ROCm dependencies
    if(NOT DEFINED TORCH_LIB_PATH)
        message(FATAL_ERROR "TORCH_LIB_PATH must be set for ROCm builds")
    endif()
    target_link_libraries(aot_bridge
        PUBLIC
            "${TORCH_LIB_PATH}/libtorch.so"
            "${TORCH_LIB_PATH}/libc10.so"
            "${TORCH_LIB_PATH}/libtorch_cpu.so"
            "${TORCH_LIB_PATH}/libtorch_hip.so"
            "${TORCH_LIB_PATH}/libc10_hip.so"
    )
else()
    target_link_libraries(aot_bridge
        PUBLIC
            ${TORCH_LIBRARIES}
    )
endif()

# Set output directory and RPATH
# TORCH_LIB_PATH is passed from build.rs for the build machine
# $ORIGIN allows portable deployment when libraries are bundled
if(DEFINED TORCH_LIB_PATH)
    set(BRIDGE_RPATH "$ORIGIN;${TORCH_LIB_PATH}")
else()
    set(BRIDGE_RPATH "$ORIGIN")
endif()

set_target_properties(aot_bridge PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    BUILD_RPATH "${BRIDGE_RPATH}"
    INSTALL_RPATH "${BRIDGE_RPATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install rules
install(TARGETS aot_bridge
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
)

install(FILES include/aot_bridge.h
    DESTINATION include
)

# Print configuration summary
message(STATUS "=== aot_bridge Configuration ===")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
