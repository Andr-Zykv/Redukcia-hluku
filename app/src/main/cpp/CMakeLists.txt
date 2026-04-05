cmake_minimum_required(VERSION 3.22.1)
project("voiceFilter")

include(FetchContent)

# FFTW3 build options — must be set before FetchContent_MakeAvailable
set(DISABLE_FORTRAN   ON  CACHE BOOL "" FORCE)  # no Fortran compiler in NDK
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)  # static lib, simpler for Android
set(BUILD_TESTS       OFF CACHE BOOL "" FORCE)
set(ENABLE_THREADS    OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENMP     OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    fftw3
    URL https://fftw.org/fftw-3.3.10.tar.gz
)

FetchContent_MakeAvailable(fftw3)

add_library(voiceFilter SHARED voicefilter.cpp)

find_library(log-lib log)

# FFTW3's CMake target does not propagate INTERFACE_INCLUDE_DIRECTORIES
target_include_directories(voiceFilter PRIVATE ${fftw3_SOURCE_DIR}/api)
target_link_libraries(voiceFilter PRIVATE ${log-lib} fftw3)
