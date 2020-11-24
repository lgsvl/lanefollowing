#!/bin/bash

set -e

LANE_FOLLOWING_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

unset ROS_DISTRO
source "/opt/ros/${ROS2_DISTRO}/setup.bash"
source "/opt/ros2-lgsvl-bridge/install/setup.bash"
source "${LANE_FOLLOWING_ROOT_DIR}/ros2_ws/install/local_setup.bash"

ros2 run lane_following drive __params:=${LANE_FOLLOWING_ROOT_DIR}/ros2_ws/src/lane_following/params/drive_visual_params.yaml &

while true; do lgsvl_bridge; done
