cmake_minimum_required(VERSION 3.10)

project(AStarPathFinder
    VERSION 0.1
    DESCRIPTION "A* path finding algorithm"
    LANGUAGES CXX
)

find_package(Eigen3 REQUIRED)

add_library(astar_path_finder SHARED src/astar_path_finder.cpp)

target_include_directories(astar_path_finder PRIVATE include)

target_link_libraries(astar_path_finder Eigen3::Eigen)
