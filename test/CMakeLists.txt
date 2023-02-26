project(testBsort)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} gtest.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ../bsort/include)
target_link_libraries(${PROJECT_NAME} gtest gmock pthread)
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)


add_definitions(-DPROJECT_DIR_PATH=\"${CMAKE_SOURCE_DIR}\")

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${PROJECT_NAME})