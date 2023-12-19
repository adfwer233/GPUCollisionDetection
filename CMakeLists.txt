cmake_minimum_required(VERSION 3.26)

project(GPUCollisionDetection)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(fetch-glfw)
include(fetch-glad)
include(fetch-glm)

add_executable(demo src/main.cpp)

add_subdirectory(src/collision_detection)

target_include_directories(demo PUBLIC include)
target_link_libraries(demo glfw glad)