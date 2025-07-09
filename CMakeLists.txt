cmake_minimum_required(VERSION 3.14)
project(MASP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0.9.9.8
)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
)

FetchContent_MakeAvailable(glm stb)

add_executable(pathtracer main.cpp)

target_link_libraries(pathtracer PRIVATE glm::glm)
target_include_directories(pathtracer PRIVATE ${stb_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pathtracer PRIVATE -O3 -march=native -fopenmp)
    target_link_libraries(pathtracer PRIVATE pthread)
endif()


find_package(OpenMP REQUIRED)
target_link_libraries(pathtracer PRIVATE OpenMP::OpenMP_CXX)

if(MINGW)
    target_link_libraries(pathtracer PRIVATE gomp)
endif()