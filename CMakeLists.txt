cmake_minimum_required(VERSION 3.16)

project(meshsearch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MESHSEARCH_BUILD_TESTS "Build the C++ test suite" ON)
option(MESHSEARCH_BUILD_EXAMPLES "Build the example" ON)
option(MESHSEARCH_BUILD_PYTHON "Build the Python bindings" OFF)

add_library(meshsearch STATIC src/MeshGrid.cpp)

target_include_directories(meshsearch PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meshsearch PUBLIC cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(meshsearch PRIVATE -Wall -Wextra)
endif()

if (MESHSEARCH_BUILD_EXAMPLES)
  add_executable(example examples/example.cpp)
  target_link_libraries(example PRIVATE meshsearch)

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(example PRIVATE -Wall -Wextra)
  endif()
endif()

if (MESHSEARCH_BUILD_TESTS)
  include(CTest)
  enable_testing()

  add_executable(test_meshgrid tests/test_meshgrid.cpp)
  target_link_libraries(test_meshgrid PRIVATE meshsearch)

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(test_meshgrid PRIVATE -Wall -Wextra)
  endif()

  add_test(NAME meshgrid COMMAND test_meshgrid)
endif()

if (MESHSEARCH_BUILD_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)

  # The module is named meshsearch; the target cannot be, the library is.
  nanobind_add_module(meshsearch_python NB_STATIC python/meshsearch_python.cpp)
  set_target_properties(meshsearch_python PROPERTIES OUTPUT_NAME meshsearch)
  target_link_libraries(meshsearch_python PRIVATE meshsearch)

  install(TARGETS meshsearch_python LIBRARY DESTINATION .)
endif()
