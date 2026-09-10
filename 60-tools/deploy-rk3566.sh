#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/10-source/rk3566/microduck"
SSH_CONFIG="$ROOT/40-deploy/rk3566/ssh/config"
KNOWN_HOSTS="$ROOT/40-deploy/rk3566/ssh/known_hosts"
REVISION="$(git -C "$REPO" rev-parse HEAD)"

exec ssh -F "$SSH_CONFIG" -o "UserKnownHostsFile=$KNOWN_HOSTS" xrange-rk3566-1 \
    "DUCK_REVISION='$REVISION' /home/xduck1/xrange/60-tools/deploy-rk3566-microduck.sh"
