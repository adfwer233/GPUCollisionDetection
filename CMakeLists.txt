cmake_minimum_required(VERSION 3.26)

project(GPUCollisionDetection)

set(ENV{http_proxy} "http://127.0.0.1:10809")
set(ENV{https_proxy} "http://127.0.0.1:10809")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(fetch-glfw)
include(fetch-glad)
include(fetch-glm)

add_executable(demo src/main.cpp)

target_include_directories(demo PUBLIC include)
target_link_libraries(demo glfw glad)