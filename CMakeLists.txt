cmake_minimum_required(VERSION 3.26)

project(GPUCollisionDetection)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(fetch-glfw)
include(fetch-glad)
include(fetch-glm)
include(fetch-gtest)

add_executable(demo src/main.cpp)

add_subdirectory(src/collision_detection)
add_subdirectory(test)

target_link_libraries(demo glfw glad glm gtest)

target_link_libraries(test collision_detection)