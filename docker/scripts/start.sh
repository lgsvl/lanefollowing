#!/usr/bin/env bash
LANE_FOLLOWING_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
nvidia-docker run -it \
	-d \
	--privileged \
	--name lanefollowing \
	--net host \
    -v $LANE_FOLLOWING_ROOT_DIR:/lanefollowing \
    lgsvl/lanefollowing:latest \
	/bin/bash
