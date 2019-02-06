#!/usr/bin/env bash
set -e

unset ROS_DISTRO
source "/opt/ros/$ROS2_DISTRO/setup.bash"

rosbridge
