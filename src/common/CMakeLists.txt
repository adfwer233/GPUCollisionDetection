project(common VERSION 1.0 DESCRIPTION "common library for collision detection" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(SRC_FILE solid/solid.cpp
        simulation/simulation.cpp
        simulation/field.cpp
        mesh/mesh.cpp
        shader/shader.cpp
        camera/camera.cpp
        solid/ground.cpp)

add_library(${PROJECT_NAME} ${SRC_FILE})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} glm glfw glad)