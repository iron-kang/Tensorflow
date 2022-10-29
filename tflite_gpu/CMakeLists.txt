cmake_minimum_required(VERSION 3.16)
project(tflite_gpu)

option(ARM "Use arm platform" OFF)
option(AARCH64 "Use arm platform" OFF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(PkgConfig)

PKG_CHECK_MODULES(GLES glesv2)
PKG_CHECK_MODULES(EGL egl)

# TensorFlow Lite Integration
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/include/)
add_executable(${PROJECT_NAME} main.cpp)

if (AARCH64)
    set(TFL_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/aarch64/libtensorflowlite.so)
    set(TFL_GPU_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/aarch64/libtensorflowlite_gpu_delegate.so)
elseif(ARM)
    set(TFL_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/arm/libtensorflowlite.so)
    set(TFL_GPU_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/arm/libtensorflowlite_gpu_delegate.so)
else()
    set(TFL_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/x86/libtensorflowlite.so)
    set(TFL_GPU_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/x86/libtensorflowlite_gpu_delegate.so)
endif()

target_link_libraries(${PROJECT_NAME} 
                      ${TFL_LIB}
                      ${TFL_GPU_LIB}
                      ${GLES_LIBRARIES}
                      ${EGL_LIBRARIES}
                      ${OpenCV_LIBS})

