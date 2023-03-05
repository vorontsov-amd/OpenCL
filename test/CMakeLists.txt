project(testBsort)

add_executable(${PROJECT_NAME} gtest.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ../bsort/include)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${OpenCLLibs})


add_definitions(-DPROJECT_DIR_PATH=\"${CMAKE_SOURCE_DIR}\")

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${PROJECT_NAME})