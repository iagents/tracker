cmake_minimum_required(VERSION 2.8)

project(kcf)

find_package(OpenCV REQUIRED)

set(CMAKE_PREFIX_PATH $ENV{HOME}/Qt5.7.0/5.7/gcc_64)

#find_package(Qt5 REQUIRED COMPONENTS Gui Widgets Test Concurrent OpenGL)

find_package(Boost 1.54 COMPONENTS filesystem)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(include)

#include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")

add_executable( kcf ${sourcefiles} )

target_link_libraries( kcf ${OpenCV_LIBS} ${Boost_LIBRARIES})




