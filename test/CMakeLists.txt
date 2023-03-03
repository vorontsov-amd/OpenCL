project(testBsort)

add_executable(${PROJECT_NAME} gtest.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ../bsort/include)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})




find_package(OpenCLICDLoader)
if (NOT OpenCLICDLoader_FOUND)
        message("If cmake couldn't find OpenCL loader, please set the path to OpenCLICDLoaderConfig.cmake in CMAKE_PREFIX_PATH")
endif()
 
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)
set(OpenCL_libs OpenCL::OpenCL Threads::Threads)
 
find_package(OpenCLHeadersCpp)
if (NOT OpenCLHeadersCpp_FOUND)
        message("If cmake couldn't find OpenCL C++ header files, please set the path to OpenCLHeadersCppConfig.cmake in CMAKE_PREFIX_PATH")
else()
        set(OpenCL_libs ${OpenCL_libs} OpenCL::HeadersCpp)
endif()
 
find_package(OpenCLHeaders)
if (NOT OpenCLHeaders_FOUND)
        message("If cmake couldn't find OpenCL header files, please set the path to OpenCLHeadersConfig.cmake in CMAKE_PREFIX_PATH")
else()
        set(OpenCL_libs ${OpenCL_libs} OpenCL::Headers)
endif()

target_link_libraries(${PROJECT_NAME} ${OpenCL_libs})




add_definitions(-DPROJECT_DIR_PATH=\"${CMAKE_SOURCE_DIR}\")

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${PROJECT_NAME})