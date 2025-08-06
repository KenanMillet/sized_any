# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Note from the author: This file was adapted from The Beman Project.
# For more information, see the comment block at the top of the root CMakeLists.txt file.

include(FetchContent)
# Fetch Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.2
    EXCLUDE_FROM_ALL
)
set(BENCHMARK_ENABLE_TESTING OFF)
# Fetch GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    EXCLUDE_FROM_ALL
)
set(INSTALL_GTEST OFF) # Disable GoogleTest installation
FetchContent_MakeAvailable(benchmark googletest)

include(GoogleTest)

# Note: The two functions below have had their names changed from beman_* to kmillet_*
#       in order to avoid overwriting the original Beman Project functions.
function(kmillet_add_benchmark name)
    kmillet_add_executable(
        TARGET ${name}
        SOURCES ${name}.benchmark.cpp ${ARGN}
        CATEGORY benchmarks
        LIBRARIES benchmark::benchmark
    )
endfunction()

function(kmillet_add_tests)
    foreach(name ${ARGN})
        kmillet_add_executable(
            TARGET ${name}
            SOURCES ${name}.test.cpp
            CATEGORY tests
            LIBRARIES GTest::gtest GTest::gtest_main
        )
        gtest_discover_tests(kmillet.sized_any.tests.${name})
    endforeach()
endfunction()

kmillet_add_tests(
    sized_any
)