# Includes the meshsearch source tree as a subproject. Configure with
# -DMESHSEARCH_SOURCE_DIR=<the meshsearch checkout> and no CMAKE_BUILD_TYPE:
# as a subproject meshsearch must build neither its tests nor its example, and
# must leave the build type to the parent, so it has to stay empty here.

cmake_minimum_required(VERSION 3.16)

project(meshsearch_subproject_consumer LANGUAGES CXX)

add_subdirectory(${MESHSEARCH_SOURCE_DIR} meshsearch)

if (TARGET test_meshgrid OR TARGET example)
  message(FATAL_ERROR "meshsearch built its tests or example as a subproject")
endif()

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(FATAL_ERROR "meshsearch set CMAKE_BUILD_TYPE to '${CMAKE_BUILD_TYPE}' as a subproject")
endif()

add_executable(consumer ../consumer.cpp)
target_link_libraries(consumer PRIVATE meshsearch::meshsearch)
