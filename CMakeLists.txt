cmake_minimum_required(VERSION 3.16)

project(meshsearch VERSION 0.2.1 LANGUAGES CXX)

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
# -fno-math-errno lets sqrt compile to the instruction, without a libm call
# kept for errno; results are unchanged.
set(MESHSEARCH_PRIVATE_OPTIONS "")
if (NOT MSVC)
  list(APPEND MESHSEARCH_PRIVATE_OPTIONS -ffp-contract=off -fno-math-errno)
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  list(APPEND MESHSEARCH_PRIVATE_OPTIONS -Wall -Wextra)
endif()

add_library(meshsearch STATIC src/MeshGrid.cpp src/MeshGridBatch.cpp)
add_library(meshsearch::meshsearch ALIAS meshsearch)

# The archive gets linked into the Python module, which is a shared object.
# Without this, GCC emits absolute relocations that a shared object cannot use
# and the link fails on Linux; Clang on macOS is position-independent anyway.
set_target_properties(meshsearch PROPERTIES POSITION_INDEPENDENT_CODE ON)

# meshsearch/Version.h, with MESHSEARCH_VERSION and its parts, comes from the
# project version above, the one place the version is written for C++. It is
# generated into the build tree, so it never goes stale in the source tree.
set(MESHSEARCH_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(cmake/Version.h.in ${MESHSEARCH_GENERATED_DIR}/meshsearch/Version.h @ONLY)

target_include_directories(meshsearch PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${MESHSEARCH_GENERATED_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(meshsearch PUBLIC cxx_std_17)
target_compile_options(meshsearch PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})

if (MESHSEARCH_BUILD_EXAMPLES)
  add_executable(example examples/example.cpp)
  target_link_libraries(example PRIVATE meshsearch)
  target_compile_options(example PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})

  add_executable(example_periodic examples/example_periodic.cpp)
  target_link_libraries(example_periodic PRIVATE meshsearch)
  target_compile_options(example_periodic PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})

  add_executable(example_batch examples/example_batch.cpp)
  target_link_libraries(example_batch PRIVATE meshsearch)
  target_compile_options(example_batch PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})
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

  add_executable(test_periodic tests/test_periodic.cpp)
  target_link_libraries(test_periodic PRIVATE meshsearch)
  target_compile_options(test_periodic PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})

  add_test(NAME periodic COMMAND test_periodic)

  add_executable(test_api tests/test_api.cpp)
  target_link_libraries(test_api PRIVATE meshsearch)
  target_compile_options(test_api PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})

  add_test(NAME api COMMAND test_api)

  # The forms test queries one grid from several threads.
  find_package(Threads REQUIRED)
  add_executable(test_forms tests/test_forms.cpp)
  target_link_libraries(test_forms PRIVATE meshsearch Threads::Threads)
  target_compile_options(test_forms PRIVATE ${MESHSEARCH_PRIVATE_OPTIONS})

  add_test(NAME forms COMMAND test_forms)

  if (MESHSEARCH_BUILD_EXAMPLES)
    add_test(NAME example_periodic COMMAND example_periodic)
    add_test(NAME example_batch COMMAND example_batch)
  endif()
endif()

if (MESHSEARCH_BUILD_PYTHON)
  # The library is linked into the module only, so none of its symbols need
  # to be visible outside it. Set here, not globally, so that a plain C++
  # build keeps default visibility for consumers of the installed archive.
  set_target_properties(meshsearch PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)

  # The module is named meshsearch; the target cannot be, the library is.
  nanobind_add_module(meshsearch_python NB_STATIC FREE_THREADED python/meshsearch_python.cpp)
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
  install(FILES ${MESHSEARCH_GENERATED_DIR}/meshsearch/Version.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/meshsearch)

  install(EXPORT meshsearchTargets
          NAMESPACE meshsearch::
          DESTINATION ${MESHSEARCH_CMAKE_DIR})

  configure_package_config_file(cmake/meshsearchConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/meshsearchConfig.cmake
    INSTALL_DESTINATION ${MESHSEARCH_CMAKE_DIR})

  # From 1.0 a request for major version N accepts any N.x.y. Before it a minor
  # release may change the interface, so a request for 0.2 accepts any 0.2.x
  # and nothing else.
  if (PROJECT_VERSION_MAJOR EQUAL 0)
    set(MESHSEARCH_COMPATIBILITY SameMinorVersion)
  else()
    set(MESHSEARCH_COMPATIBILITY SameMajorVersion)
  endif()
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/meshsearchConfigVersion.cmake
    COMPATIBILITY ${MESHSEARCH_COMPATIBILITY})

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/meshsearchConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/meshsearchConfigVersion.cmake
          DESTINATION ${MESHSEARCH_CMAKE_DIR})
endif()
