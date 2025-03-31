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

find_package(Torch REQUIRED)

find_library(OPENVINO_BACKEND openvino_backend REQUIRED)
find_library(PORTABLE_OPS_LIB portable_ops_lib REQUIRED)
find_library(EXECUTORCH_CORE executorch_core REQUIRED)
find_library(XNNPACK_BACKEND xnnpack_backend REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

set(PROJECT_SOURCES
    main.cpp
    inference.h
    inference.cpp
    ${EXECUTORCH_ROOT}/extension/data_loader/file_data_loader.cpp
    ${EXECUTORCH_ROOT}/extension/evalue_util/print_evalue.cpp
    ${EXECUTORCH_ROOT}/extension/runner_util/inputs.cpp
    ${EXECUTORCH_ROOT}/extension/runner_util/inputs_portable.cpp
)

add_executable(Yolov8CPPInference ${PROJECT_SOURCES})
target_link_libraries(Yolov8CPPInference ${OpenCV_LIBS} ${OPENVINO_BACKEND} ${PORTABLE_OPS_LIB} gflags ${EXECUTORCH_CORE} ${XNNPACK_BACKEND})
target_include_directories(Yolov8CPPInference PUBLIC ${EXECUTORCH_ROOT}/..)
