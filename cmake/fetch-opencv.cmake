include(FetchContent)

FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
)

FetchContent_MakeAvailable(opencv)