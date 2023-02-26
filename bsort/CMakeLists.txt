project(bsort)

add_executable(${PROJECT_NAME} ./src/BitonicSorter.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ./include)

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)

set(CL_PROGRAM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/bsort.cl")
set(OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include/sourcePath.h")
configure_file(./include/sourcePath.h.in ${OUTPUT_PATH} @ONLY)

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${PROJECT_NAME})