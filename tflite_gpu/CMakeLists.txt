cmake_minimum_required(VERSION 3.16)
project(tflite_gpu)

find_package(PkgConfig)
PKG_CHECK_MODULES(OPENCV opencv4)
PKG_CHECK_MODULES(GLES glesv2)
PKG_CHECK_MODULES(EGL egl)

#include_directories(${OpenCV_INCLUDE_DIRS})
# TensorFlow Lite Integration
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/include/
		    ${OpenCV_INCLUDE_DIRS})
include_directories( ${OPENCV_INCLUDE_DIRS} )
add_executable(${PROJECT_NAME} main.cpp)

IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(TFL_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/aarch64/libtensorflowlite.so)
    set(TFL_GPU_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/aarch64/libtensorflowlite_gpu_delegate.so)
ELSEIF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    set(TFL_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/armhf/libtensorflowlite.so)
    set(TFL_GPU_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-src/libs/armhf/libtensorflowlite_gpu_delegate.so)
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
		      ${OPENCV_LIBRARIES})

