#!/bin/zsh
set -eu
XDUCK_VERSION_DIR="$(cd "$(dirname "$0")" && pwd)"
XDUCK_REPO_DIR="$(cd "$XDUCK_VERSION_DIR/../../.." && pwd)"
if command -v mjpython >/dev/null 2>&1; then
  XDUCK_MJPYTHON="$(command -v mjpython)"
else
  XDUCK_MJPYTHON="$(python3 -m site --user-base)/bin/mjpython"
fi
exec "$XDUCK_MJPYTHON" "$XDUCK_REPO_DIR/mujoco_viewer.py" "$XDUCK_VERSION_DIR/mjcf" --keyframe 0 --paused
