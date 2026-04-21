cmake_minimum_required(VERSION 3.18)

project(didactic-tensor-compiler)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json REQUIRED)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# nanobind (installed via pip)
execute_process(
    COMMAND python3 -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${NANOBIND_CMAKE_DIR})
find_package(nanobind CONFIG REQUIRED)
find_package(CUDA)

if (CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    # for older compilers and cmake
    # set(CMAKE_CUDA_ARCHITECTURES 86)
    message("Building with GPU support")
    add_library(gpu_exec src/gpu_exec.cu include/gpu_exec.h)
    target_link_libraries(gpu_exec PRIVATE cublas)
    set(OPS_GPU_LIB gpu_exec)
    add_compile_definitions(CUDA_FOUND)
    set(OPS_METAL_LIB "")
elseif (APPLE)
    message("Building with Metal support (M1)")

    set(METAL_SRC ${CMAKE_SOURCE_DIR}/src/shaders.metal)

    # Detect whether the full Xcode metal compiler is available
    execute_process(
        COMMAND xcrun --find metal
        OUTPUT_VARIABLE METAL_COMPILER
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    add_library(metal_exec src/metal_exec.cpp include/metal_exec.h)
    target_include_directories(metal_exec PRIVATE metal-cpp)
    target_link_libraries(metal_exec PRIVATE "-framework Metal" "-framework Foundation" "-framework QuartzCore")
    # Always tell the binary where the .metal source lives (used by the fallback path)
    target_compile_definitions(metal_exec PRIVATE SHADER_SRC_PATH="${METAL_SRC}")

    if(METAL_COMPILER)
        message("  xcrun metal found — shaders will be compiled at build time")
        set(METAL_AIR ${CMAKE_BINARY_DIR}/shaders.air)
        set(METAL_LIB ${CMAKE_SOURCE_DIR}/python/shaders.metallib)
        add_custom_command(
            OUTPUT  ${METAL_LIB}
            COMMAND xcrun -sdk macosx metal  -c ${METAL_SRC} -o ${METAL_AIR}
            COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
            DEPENDS ${METAL_SRC}
            COMMENT "Compiling Metal shaders"
        )
        add_custom_target(metal_shaders DEPENDS ${METAL_LIB})
        add_dependencies(metal_exec metal_shaders)
        target_compile_definitions(metal_exec PRIVATE METALLIB_PATH="${METAL_LIB}")
    else()
        message("  xcrun metal not found — shaders will compile at runtime from source")
    endif()

    set(OPS_GPU_LIB "")
    set(OPS_METAL_LIB metal_exec)
    add_compile_definitions(METAL_FOUND)
else()
    message("Building with CPU support")
    set(OPS_GPU_LIB "")
    set(OPS_METAL_LIB "")
endif()

add_library(autotuner src/autotuner.cpp include/autotuner.h)
add_library(mnist src/mnist.cpp include/mnist.h)
add_library(frontend src/frontend.cpp include/frontend.h)
add_library(optimizers src/optimizers.cpp include/optimizers.h)
add_library(ops src/ops.cpp include/ops.h)
add_library(types src/types.cpp include/types.h)
add_library(passes src/passes.cpp include/passes.h)
add_executable(main src/main.cpp)

target_link_libraries(frontend nlohmann_json::nlohmann_json)
target_link_libraries(passes nlohmann_json::nlohmann_json)
target_link_libraries(optimizers nlohmann_json::nlohmann_json)
target_link_libraries(main mnist autotuner optimizers frontend types ops passes ${OPS_GPU_LIB} ${OPS_METAL_LIB} nlohmann_json::nlohmann_json)

# Python extension module
nanobind_add_module(tensor_frontend src/python_binding.cpp)
target_link_libraries(tensor_frontend PRIVATE mnist autotuner optimizers frontend types ops passes ${OPS_GPU_LIB} ${OPS_METAL_LIB} nlohmann_json::nlohmann_json)
target_include_directories(tensor_frontend PRIVATE metal-cpp)
add_custom_command(TARGET tensor_frontend POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tensor_frontend> ${CMAKE_SOURCE_DIR}/python/
)