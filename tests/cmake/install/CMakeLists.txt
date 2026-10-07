# Finds meshsearch as an installed package. Configure with
# -DCMAKE_PREFIX_PATH=<the prefix meshsearch was installed to>.

cmake_minimum_required(VERSION 3.16)

project(meshsearch_install_consumer LANGUAGES CXX)

find_package(meshsearch 0.2 CONFIG REQUIRED)

add_executable(consumer ../consumer.cpp)
target_link_libraries(consumer PRIVATE meshsearch::meshsearch)
