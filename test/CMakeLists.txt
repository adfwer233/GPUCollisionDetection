project(test CUDA)

enable_testing()

add_executable(test collision_detection/sort_test.cpp)

target_link_libraries(test GTest::gtest_main)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
