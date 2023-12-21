project(test CUDA)

set(CMAKE_CXX_STANDARD 20)

enable_testing()

add_executable(test collision_detection/sort_test.cu
        collision_detection/scan_test.cu)

target_link_libraries(test GTest::gtest_main)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
