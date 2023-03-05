cmake_minimum_required(VERSION 3.5)

project(Bsort)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wpedantic")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_TYPE RELEASE)

set(TYPE "int" CACHE STRING "Type to compile with")
message(STATUS "Using type = ${TYPE}")
add_compile_definitions(TYPE=${TYPE})


set(PLATFORM "ANY_PLATFORM" CACHE STRING "Computing platform")
message(STATUS "Using platform = ${PLATFORM}")
add_compile_definitions(PLATFORM=${PLATFORM})


add_subdirectory(bsort)
# add_subdirectory(test)