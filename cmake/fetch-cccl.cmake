#include(FetchContent)
#
#FetchContent_Declare(
#        cccl
#        GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
#        GIT_TAG v2.2.0
#)
#
#FetchContent_MakeAvailable(cccl)

set(CCCL_REPOSITORY "nvidia/cccl" CACHE STRING "GitHub repository to fetch CCCL from")
set(CCCL_TAG "main" CACHE STRING "Git tag/branch to fetch from CCCL repository")

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY ${CCCL_REPOSITORY}
        GIT_TAG ${CCCL_TAG}
)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()