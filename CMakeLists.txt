cmake_minimum_required(VERSION 3.10)
project(MergeJson VERSION 1.0
                    DESCRIPTION "SimPassFinder"
                    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build (Debug or Release)" FORCE)
endif()

file(GLOB_RECURSE SOURCES cpp/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

# target_include_directories(${PROJECT_NAME} PRIVATE include/)
# target_link_libraries(${PROJECT_NAME} PRIVATE some_library)
