project(collision_detection VERSION 1.0 DESCRIPTION "project for learning cuda" LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

set(SRC_FILE main.cu)

add_executable(${PROJECT_NAME} ${SRC_FILE})
