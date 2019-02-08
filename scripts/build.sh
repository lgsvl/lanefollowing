#!/bin/bash

set -e

unset ROS_DISTRO
source "/opt/ros/crystal/setup.bash"

LANE_FOLLOWING_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd $LANE_FOLLOWING_ROOT_DIR/ros2_ws
colcon build --symlink-install
