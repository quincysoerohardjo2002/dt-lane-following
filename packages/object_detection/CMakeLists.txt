cmake_minimum_required(VERSION 2.8.3)
project(object_detection)

find_package(catkin REQUIRED COMPONENTS
  duckietown
  duckietown_msgs
  rospy
  sensor_msgs
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
