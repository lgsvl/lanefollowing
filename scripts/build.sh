#!/bin/bash

set -e

LANE_FOLLOWING_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

unset ROS_DISTRO
source "/opt/ros/crystal/setup.bash"

cd $LANE_FOLLOWING_ROOT_DIR/ros2_ws
colcon build --symlink-install
