#!/usr/bin/env bash
set -e

unset ROS_DISTRO
source "/opt/ros/$ROS1_DISTRO/setup.bash"
source "/lanefollowing/catkin_ws/devel/setup.bash"

roslaunch rosbridge_server rosbridge_websocket.launch
