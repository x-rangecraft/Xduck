#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
PROJECT="$ROOT/10-source/tools/motor/gf43x40-10-i2rt"
ENV_DIR="$ROOT/20-build/tools/motor/gf43x40-10-i2rt/.venv"

mkdir -p "$(dirname "$ENV_DIR")"
if [ ! -x "$ENV_DIR/bin/python" ]; then
    python3 -m venv "$ENV_DIR"
fi
if [ ! -L "$PROJECT/.venv" ]; then
    test ! -e "$PROJECT/.venv" || { echo "error: $PROJECT/.venv exists and is not a link" >&2; exit 1; }
    ln -s ../../../../20-build/tools/motor/gf43x40-10-i2rt/.venv "$PROJECT/.venv"
fi
"$ENV_DIR/bin/python" -m pip install -r "$PROJECT/requirements.txt"
