#!/bin/bash

set -e

unset ROS_DISTRO
source "/opt/ros/crystal/setup.bash"
source "/lanefollowing/ros2_ws/install/local_setup.bash"

ros2 run lane_following collect &

while true; do rosbridge; done
