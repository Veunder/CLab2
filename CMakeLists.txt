cmake_minimum_required(VERSION 4.3)
project(CLab2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)

add_executable(CLab2 main.cpp)

target_link_libraries(CLab2 PRIVATE ${OpenCV_LIBS})

target_include_directories(CLab2 PRIVATE ${OpenCV_INCLUDE_DIRS})