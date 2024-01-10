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

include(CPM)
include(fetch-cccl)

add_subdirectory(src/demo)
add_subdirectory(src/collision_detection)
add_subdirectory(src/common)
add_subdirectory(test)

target_link_libraries(demo glfw glad glm gtest)

target_link_libraries(test collision_detection)

message(STATUS "Build documentation open")
include(build-doxygen)
build_doxygen()

include(clang-format)