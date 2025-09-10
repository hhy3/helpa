include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark's own tests")
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
FetchContent_MakeAvailable(benchmark)

add_executable(bench_single bench_single.cc)
target_link_libraries(bench_single helpa benchmark::benchmark benchmark::benchmark_main)
