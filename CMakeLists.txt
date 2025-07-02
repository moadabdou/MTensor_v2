
cmake_minimum_required(VERSION 4.0.2)

project(MTensor VERSION 2.0 LANGUAGES CXX)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

include(FetchContent)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/config)

include(CompilerFlags)
include(GoogletestConfig)
include(oneDNNConfig)
include(OpenMPConfig)

add_subdirectory(src)
add_subdirectory(tests)





