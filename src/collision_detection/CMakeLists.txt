project(collision_detection VERSION 1.0 DESCRIPTION "project for learning cuda" LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

set(SRC_FILE main.cu
        gpu_scan.cu
        cpu/cpu_collision_detection.cpp
        cpu/cpu_sweep_and_prune_collision_detection.cpp
        gpu/gpu_sweep_and_prune_collision_detection.cu
        gpu/gpu_sweep_and_prune_collision_detection_kernel.cuh)

add_library(${PROJECT_NAME} ${SRC_FILE})

target_link_libraries(${PROJECT_NAME} glm common)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
