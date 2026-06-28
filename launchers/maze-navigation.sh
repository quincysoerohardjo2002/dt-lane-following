#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# Start maze navigatie: Dijkstra path planning + navigator state machine
# + YOLO duckie detectie + ASCII dashboard
dt-exec roslaunch --wait maze_navigation maze_navigation.launch

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
