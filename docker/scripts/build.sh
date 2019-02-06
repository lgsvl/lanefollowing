#!/usr/bin/env bash
LANE_FOLLOWING_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
docker build -t lgsvl/lanefollowing:latest $LANE_FOLLOWING_ROOT_DIR/docker/
