include(FetchContent)

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG ${GLAD_GIT_TAG})

FetchContent_MakeAvailable(glad)