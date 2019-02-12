#!/bin/bash

set -e

LANE_FOLLOWING_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

unset ROS_DISTRO
source "/opt/ros/crystal/setup.bash"
source "$LANE_FOLLOWING_ROOT_DIR/ros2_ws/install/local_setup.bash"

ros2 run lane_following collect &

while true; do rosbridge; done
