project(demo VERSION 1.0 DESCRIPTION "Collision detection demo" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(SRC_FILE main.cpp)

add_definitions(-DSHADER_DIR="./shader")

add_definitions(-DOPENCV_EXPORT)

add_executable(${PROJECT_NAME} ${SRC_FILE})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} glm glfw glad common collision_detection)

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/src/shader" $<TARGET_FILE_DIR:${PROJECT_NAME}>/shader
)