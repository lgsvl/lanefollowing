#!/usr/bin/env bash
set -e

LANE_FOLLOWING_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

unset ROS_DISTRO
source "/opt/ros/$ROS1_DISTRO/setup.bash"
source "$LANE_FOLLOWING_ROOT_DIR/catkin_ws/devel/setup.bash"

roslaunch rosbridge_server rosbridge_websocket.launch
