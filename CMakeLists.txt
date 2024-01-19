cmake_minimum_required(VERSION 3.26)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(GPUCollisionDetection)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(fetch-glfw)
include(fetch-glad)
include(fetch-glm)
include(fetch-gtest)
#include(fetch-opencv)

#include(CPM)
#include(fetch-cccl)

set(EXPORT_VIDEO "OFF" CACHE BOOL "Whether export video")

find_package( OpenCV )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(src/demo)
add_subdirectory(src/collision_detection)
add_subdirectory(src/common)
add_subdirectory(test)

if(OpenCV_FOUND)
    message(STATUS "found opencv")
    target_link_libraries(demo glfw glad glm gtest ${OpenCV_LIBS})
else()
    target_link_libraries(demo glfw glad glm gtest)
endif()


target_link_libraries(test collision_detection)

message(STATUS "Build documentation open")
include(build-doxygen)
build_doxygen()

include(clang-format)