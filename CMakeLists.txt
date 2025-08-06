# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#[[
Note from the author:

This file and the other CMakeLists.txt files within this repository were adapted from
The Beman Project. Specifically, various CMakeLists blocks are being copied and altered
to ensure that my CMakeLists.txt files are consistent with the The Beman Project's standards.
I am not copying out of laziness; I am simply trying to learn to write higher quality CMake projects
while also finding the CMake language as a whole to be quite confusing.

The Beman Project: https://bemanproject.org/

In particular, the libraries that I am adapting CMakeLists.txt files from The Beman Project are:
- beman/exemplar (for its general CMake practices): https://github.com/bemanproject/exemplar
- beman/inplace_vector (for its project configuration): https://github.com/bemanproject/inplace_vector 
- beman/scope (for its simplicity as a header-only library): https://github.com/bemanproject/scope
- beman/any_view (for its separation of benchmarks and unit-tests): https://github.com/bemanproject/any_view

A lot of places in this file and the other CMakeLists.txt files within this repository
use a naming scheme of kmillet.sized_any or kmillet::sized_any. If my library is one day
adopted by The Beman Project, I will change the naming scheme from kmillet to beman, but
until then, I do not want to misrepresent beman's work.
]]

cmake_minimum_required(VERSION 3.25)

# [CMAKE.PROJECT_NAME]
project(
    kmillet.sized_any
    DESCRIPTION "A stack-optimized drop-in replacement for std::any"
    LANGUAGES CXX
    VERSION 0.0.1
)

# [CMAKE.SKIP_TESTS]
option(
    KMILLET_SIZED_ANY_BUILD_TESTS
    "Enable building tests and test infrastructure. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
    ${PROJECT_IS_TOP_LEVEL}
)

# [CMAKE.SKIP_EXAMPLES]
option(
    KMILLET_SIZED_ANY_BUILD_EXAMPLES
    "Enable building examples. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
    ${PROJECT_IS_TOP_LEVEL}
)

option(
    KMILLET_SIZED_ANY_INSTALL_CONFIG_FILE_PACKAGE
    "Enable creating and installing a CMake config-file package. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
    ${PROJECT_IS_TOP_LEVEL}
)

configure_file(
    include/kmillet/sized_any/config.hpp.in
    include/kmillet/sized_any/config.hpp
    @ONLY
)

# [CMAKE.LIBRARY_NAME]
add_library(kmillet.sized_any INTERFACE)
# [CMAKE.LIBRARY_ALIAS]
add_library(kmillet::sized_any ALIAS kmillet.sized_any)

set_target_properties(
    kmillet.sized_any
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS ON
        EXPORT_NAME sized_any
)

target_sources(
    kmillet.sized_any
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS include
        FILES include/kmillet/sized_any/sized_any.hpp
)

include(GNUInstallDirs)

install(
    TARGETS kmillet.sized_any
    EXPORT kmillet.sized_any-targets
    FILE_SET HEADERS
)

if(KMILLET_SIZED_ANY_INSTALL_CONFIG_FILE_PACKAGE)
    include(CMakePackageConfigHelpers)

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/kmillet.sized_any-config-version.cmake
        COMPATIBILITY ExactVersion
    )

    install(
        FILES
            cmake/kmillet.sized_any-config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/kmillet.sized_any-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kmillet.sized_any
        COMPONENT kmillet.sized_any
    )

    install(
        EXPORT kmillet.sized_any-targets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kmillet.sized_any
        NAMESPACE kmillet::
        CXX_MODULES_DIRECTORY
        cxx-modules
        COMPONENT kmillet.sized_any
    )
endif()

# The function below is directly copied from beman.any_view/CMakeLists.txt.
# The name has been changed from beman_add_executable to kmillet_add_executable
# so that it does not overwrite the beman_add_executable function.
function(kmillet_add_executable)
    set(options)
    set(oneValueArgs CATEGORY TARGET)
    set(multiValueArgs SOURCES LIBRARIES)

    cmake_parse_arguments(
        kmillet_executable
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    # [CMAKE.TARGET_NAMES]
    set(category kmillet.sized_any.${kmillet_executable_CATEGORY})
    set(target ${category}.${kmillet_executable_TARGET})

    add_executable(${target})
    # [CMAKE.PASSIVE_PROJECTS]
    # set features on executables, not on interface library
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_sources(${target} PRIVATE ${kmillet_executable_SOURCES})
    target_link_libraries(
        ${target}
        PRIVATE kmillet::sized_any ${kmillet_executable_LIBRARIES}
    )

    if(NOT TARGET ${category})
        add_custom_target(${category})
    endif()

    add_dependencies(${category} ${target})
endfunction()

if(KMILLET_SIZED_ANY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/kmillet/sized_any)
endif()

if(KMILLET_SIZED_ANY_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()