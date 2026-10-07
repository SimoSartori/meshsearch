cmake_minimum_required(VERSION 3.16)

project(meshsearch VERSION 0.2.0 LANGUAGES CXX)

# PROJECT_IS_TOP_LEVEL needs CMake 3.21; this is the 3.16 equivalent, true
# for a standalone build and false under add_subdirectory or FetchContent.
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MESHSEARCH_IS_TOP_LEVEL ON)
else()
  set(MESHSEARCH_IS_TOP_LEVEL OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Only a top-level build may choose these for the whole build tree; as a
# subproject, the parent's choices stand.
if (MESHSEARCH_IS_TOP_LEVEL)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
endif()

# scikit-build-core defines SKBUILD. The wheel carries the Python module only,
# so the C++ install rules are off there, and stay off when a parent project's
# wheel builds meshsearch as a subproject.
if (DEFINED SKBUILD)
  set(MESHSEARCH_INSTALL_DEFAULT OFF)
else()
  set(MESHSEARCH_INSTALL_DEFAULT ON)
endif()

option(MESHSEARCH_BUILD_TESTS "Build the C++ test suite" ${MESHSEARCH_IS_TOP_LEVEL})
option(MESHSEARCH_BUILD_EXAMPLES "Build the example" ${MESHSEARCH_IS_TOP_LEVEL})
option(MESHSEARCH_BUILD_PYTHON "Build the Python bindings" OFF)
option(MESHSEARCH_INSTALL "Generate the C++ install rules" ${MESHSEARCH_INSTALL_DEFAULT})

include(GNUInstallDirs)

# No a*b+c is fused into one FMA: whether the compiler contracts depends on the
# compiler and the instruction set, so contraction would make distances, and
# with them boundary and tie decisions, depend on the platform. Private, so it
# applies to meshsearch's own translation units and reaches no consumer.
set(MESHSEARCH_PRIVATE_OPTIONS "")
if (NOT MSVC)
  list(APPEND MESHSEARCH_PRIVATE_OPTIONS -ffp-contract=off)
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  list(APPEND MESHSEARCH_PRIVATE_OPTIONS -Wall -Wextra)
endif()

add_library(meshsearch STATIC src/MeshGrid.cpp)
add_library(meshsearch::meshsearch ALIAS meshsearch)

# The archive gets linked into the Python module, which is a shared object.
# Without this, GCC emits absolute relocations that a shared object cannot use
# and the link fails on Linux; Clang on macOS is position-independent anyway.
set_target_properties(meshsearch PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(meshsearch PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(meshsearch PUBLIC cxx_std_17)
target_compile_options(meshsearch PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})

if (MESHSEARCH_BUILD_EXAMPLES)
  add_executable(example examples/example.cpp)
  target_link_libraries(example PRIVATE meshsearch)
  target_compile_options(example PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})
endif()

if (MESHSEARCH_BUILD_TESTS)
  include(CTest)
  enable_testing()

  # The brute-force reference in the test computes distances the way the
  # library does, so it must be compiled with the same contraction setting.
  add_executable(test_meshgrid tests/test_meshgrid.cpp)
  target_link_libraries(test_meshgrid PRIVATE meshsearch)
  target_compile_options(test_meshgrid PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})

  add_test(NAME meshgrid COMMAND test_meshgrid)
endif()

if (MESHSEARCH_BUILD_PYTHON)
  # The library is linked into the module only, so none of its symbols need
  # to be visible outside it. Set here, not globally, so that a plain C++
  # build keeps default visibility for consumers of the installed archive.
  set_target_properties(meshsearch PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)

  # The module is named meshsearch; the target cannot be, the library is.
  nanobind_add_module(meshsearch_python NB_STATIC python/meshsearch_python.cpp)
  set_target_properties(meshsearch_python PROPERTIES OUTPUT_NAME meshsearch)
  target_link_libraries(meshsearch_python PRIVATE meshsearch)
  target_compile_options(meshsearch_python PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})

  install(TARGETS meshsearch_python LIBRARY DESTINATION .)
endif()

if (MESHSEARCH_INSTALL)
  include(CMakePackageConfigHelpers)

  set(MESHSEARCH_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/meshsearch)

  install(TARGETS meshsearch EXPORT meshsearchTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/meshsearch DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(EXPORT meshsearchTargets
          NAMESPACE meshsearch::
          DESTINATION ${MESHSEARCH_CMAKE_DIR})

  configure_package_config_file(cmake/meshsearchConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/meshsearchConfig.cmake
    INSTALL_DESTINATION ${MESHSEARCH_CMAKE_DIR})

  # Before 1.0 a minor release may change the interface, so a request for 0.2
  # accepts any 0.2.x and nothing else.
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/meshsearchConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/meshsearchConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/meshsearchConfigVersion.cmake
          DESTINATION ${MESHSEARCH_CMAKE_DIR})
endif()
