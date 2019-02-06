#!/usr/bin/env bash
set -e

unset ROS_DISTRO
source "/opt/ros/$ROS2_DISTRO/setup.bash"
source "/lanefollowing/ros2_ws/install/local_setup.bash"

ros2 run lane_following drive
