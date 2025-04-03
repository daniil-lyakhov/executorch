cmake_minimum_required(VERSION 3.5)

project(Yolov8CPPInference VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# find `executorch` libraries Same as for gflags
set(executorch_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../lib/cmake/ExecuTorch)
find_package(executorch CONFIG REQUIRED)
target_link_options_shared_lib(executorch)

set(link_libraries gflags)
list(APPEND link_libraries portable_ops_lib portable_kernels)
target_link_options_shared_lib(portable_ops_lib)

set(xnnpack_backend_libs xnnpack_backend XNNPACK microkernels-prod)
list(APPEND link_libraries ${xnnpack_backend_libs})
target_link_options_shared_lib(xnnpack_backend)


list(APPEND link_libraries extension_threadpool pthreadpool)
list(APPEND _common_include_directories
     ${XNNPACK_ROOT}/third-party/pthreadpool/include
)

list(APPEND link_libraries extension_threadpool cpuinfo)
list(APPEND _common_include_directories
     ${XNNPACK_ROOT}/third-party/cpuinfo/include
)

message(STATUS ${link_libraries})

set(PROJECT_SOURCES
    main.cpp
    inference.h
    ${EXECUTORCH_ROOT}/extension/data_loader/file_data_loader.cpp
    ${EXECUTORCH_ROOT}/extension/evalue_util/print_evalue.cpp
    ${EXECUTORCH_ROOT}/extension/runner_util/inputs.cpp
    ${EXECUTORCH_ROOT}/extension/runner_util/inputs_portable.cpp
)

add_executable(Yolov8CPPInference ${PROJECT_SOURCES})
target_link_libraries(Yolov8CPPInference PUBLIC
    ${link_libraries}
    ${OpenCV_LIBS}
    executorch_core
    extension_module
    extension_tensor
)

find_package(Threads REQUIRED)
target_link_libraries(Yolov8CPPInference PRIVATE Threads::Threads)
target_include_directories(Yolov8CPPInference PUBLIC ${_common_include_directories})