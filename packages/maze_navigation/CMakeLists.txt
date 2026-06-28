cmake_minimum_required(VERSION 3.0.2)
project(maze_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  duckietown_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/path_planner_node.py
  src/navigator_node.py
  src/map_visualizer_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
