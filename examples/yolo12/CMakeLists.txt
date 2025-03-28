cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(yolov8_libtorch_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# -------------- OpenCV --------------
set(OpenCV_DIR "/path/to/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- libtorch --------------
list(APPEND CMAKE_PREFIX_PATH "/path/to/libtorch")
set(Torch_DIR "/path/to/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("${TORCH_LIBRARIES}")
message("${TORCH_INCLUDE_DIRS}")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET yolov8_libtorch_example
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:yolov8_libtorch_example>)
# endif (MSVC)

# -------------- openvino-backend --------------
find_library(OPENVINO_BACKEND openvino_backend  REQUIRED)
find_library(PORTABLE_OPS_LIB portable_ops_lib  REQUIRED)
find_library(EXECUTORCH_CORE executorch_core REQUIRED)
find_library(XNNPACK_BACKEND xnnpack_backend REQUIRED)


include_directories(${TORCH_INCLUDE_DIRS})

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

add_executable(yolov8_libtorch_inference
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
    ${EXECUTORCH_ROOT}/extension/data_loader/file_data_loader.cpp
    ${EXECUTORCH_ROOT}/extension/evalue_util/print_evalue.cpp
    ${EXECUTORCH_ROOT}/extension/runner_util/inputs.cpp
    ${EXECUTORCH_ROOT}/extension/runner_util/inputs_portable.cpp
)
target_link_libraries(yolov8_libtorch_inference ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${OPENVINO_BACKEND} ${PORTABLE_OPS_LIB} gflags ${EXECUTORCH_CORE} ${XNNPACK_BACKEND})
target_include_directories(yolov8_libtorch_inference PUBLIC ${EXECUTORCH_ROOT}/..)

set_property(TARGET yolov8_libtorch_inference PROPERTY CXX_STANDARD 17)
