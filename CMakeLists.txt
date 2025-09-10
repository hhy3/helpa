cmake_minimum_required (VERSION 3.15)
project(helpa LANGUAGES CXX)

add_library(helpa INTERFACE)
target_include_directories(helpa INTERFACE .) 

set(CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS  "-Wall -Wextra -O3 -std=c++17 -march=native -fpic -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )

OPTION(HELPA_USE_AVX512BF16 "Use avx512_bf16 instruction set" ON)
OPTION(HELPA_USE_AVX512FP16 "Use avx512_fp16 instruction set" OFF)

if(HELPA_USE_AVX512BF16)
    add_definitions(-DHELPA_USE_AVX512BF16)
endif(HELPA_USE_AVX512BF16)

if(HELPA_USE_AVX512FP16)
    add_definitions(-DHELPA_USE_AVX512FP16)
endif(HELPA_USE_AVX512FP16)

if(HELPA_USE_AVX512BF16)
    target_compile_definitions(helpa INTERFACE HELPA_USE_AVX512BF16)
endif()

if(HELPA_USE_AVX512FP16)
    target_compile_definitions(helpa INTERFACE HELPA_USE_AVX512FP16)
endif()


add_executable(main main.cc)
target_link_libraries(main helpa)

option(WITH_BENCH "Build benchmark" ON)
if (WITH_BENCH)
    add_subdirectory(bench)
endif()
