cmake_minimum_required(VERSION 3.16)

project(meshsearch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(meshsearch STATIC src/MeshGrid.cpp)

target_include_directories(meshsearch PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meshsearch PUBLIC cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(meshsearch PRIVATE -Wall -Wextra)
endif()

add_executable(example examples/example.cpp)
target_link_libraries(example PRIVATE meshsearch)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(example PRIVATE -Wall -Wextra)
endif()

include(CTest)
enable_testing()

add_executable(test_meshgrid tests/test_meshgrid.cpp)
target_link_libraries(test_meshgrid PRIVATE meshsearch)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(test_meshgrid PRIVATE -Wall -Wextra)
endif()

add_test(NAME meshgrid COMMAND test_meshgrid)
