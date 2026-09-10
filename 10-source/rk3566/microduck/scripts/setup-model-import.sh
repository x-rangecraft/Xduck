#!/bin/sh
# Run as part of an authorized deployment, not from the robot control loop.
# No service restart and no motor access. Requires python3-venv and network access.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_PYTHON_DIR=${1:-/var/lib/robotd/model-python}
python3 -m venv "$MODEL_PYTHON_DIR"
"$MODEL_PYTHON_DIR/bin/python" -m pip install --disable-pip-version-check -r "$SCRIPT_DIR/model-import-requirements.txt"
"$MODEL_PYTHON_DIR/bin/python" -I -c 'import torch, onnx, onnxruntime, numpy; print("model converter:", torch.__version__, onnx.__version__, onnxruntime.__version__, numpy.__version__)'
printf 'Set ROBOT_MODEL_PYTHON=%s/bin/python for robotd (default deployment path is detected automatically).\n' "$MODEL_PYTHON_DIR"
