cmake_minimum_required(VERSION 3.18)
project(c_annfund_win LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(test_compat
    tools/test_compat.c
    src/win_compat.c
    src/queue.c
)
target_include_directories(test_compat PRIVATE include)
target_compile_definitions(test_compat PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(test_compat PRIVATE ws2_32)

add_executable(test_bin_to_pth
    tools/test_bin_to_pth.c
    src/weights_pth.c
)
target_include_directories(test_bin_to_pth PRIVATE include)
target_compile_definitions(test_bin_to_pth PRIVATE _CRT_SECURE_NO_WARNINGS)

add_executable(test_pth_to_bin
    tools/test_pth_to_bin.c
    src/weights_pth.c
)
target_include_directories(test_pth_to_bin PRIVATE include)
target_compile_definitions(test_pth_to_bin PRIVATE _CRT_SECURE_NO_WARNINGS)

if(NOT CMAKE_CUDA_COMPILER AND WIN32)
    file(GLOB _ann_nvcc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/nvcc.exe")
    if(_ann_nvcc)
        list(SORT _ann_nvcc COMPARE NATURAL)
        list(GET _ann_nvcc -1 CMAKE_CUDA_COMPILER)
        get_filename_component(_ann_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(CUDAToolkit_ROOT "${_ann_cuda_bin}" DIRECTORY)
        set(ENV{CUDA_PATH} "${CUDAToolkit_ROOT}")
        message(STATUS "Using nvcc: ${CMAKE_CUDA_COMPILER}")
    endif()
    unset(_ann_nvcc)
    unset(_ann_cuda_bin)
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING
        "CUDA Toolkit (nvcc) not found; skipping annfund. "
        "Install CUDA Toolkit 12.4+ with sm_75, then re-run cmake.")
    return()
endif()

if(CMAKE_CUDA_COMPILER AND NOT CUDAToolkit_ROOT)
    get_filename_component(_ann_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(CUDAToolkit_ROOT "${_ann_cuda_bin}" DIRECTORY)
    unset(_ann_cuda_bin)
endif()
if(CUDAToolkit_ROOT)
    set(ENV{CUDA_PATH} "${CUDAToolkit_ROOT}")
    set(CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR "${CUDAToolkit_ROOT}/")
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

add_executable(annfund
    src/main.c
    src/db.c
    src/dataset.c
    src/queue.c
    src/weights.c
    src/weights_pth.c
    src/win_compat.c
    src/model.cu
    third_party/sqlite/sqlite3.c
)
target_include_directories(annfund PRIVATE include third_party/sqlite)
target_compile_definitions(annfund PRIVATE
    _CRT_SECURE_NO_WARNINGS
    SQLITE_THREADSAFE=1
)
target_link_libraries(annfund PRIVATE CUDA::cudart CUDA::cublas ws2_32)
set_target_properties(annfund PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)
