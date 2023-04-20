cmake_minimum_required(VERSION 3.0.2)
project(convex_optimize_fitting)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  tf
  obsdet_msgs
)

find_package(OpenCV REQUIRED)
find_package (Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")

catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS pcl_ros obsdet_msgs
)

link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

set(SOURCE_FILES
  src/convex_optimize_fitting.cpp
)


add_executable(convex_optimize_fitting ${SOURCE_FILES})

add_dependencies(convex_optimize_fitting
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(convex_optimize_fitting
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)