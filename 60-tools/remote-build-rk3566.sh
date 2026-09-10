#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
SSH_CONFIG="$ROOT/40-deploy/rk3566/ssh/config"
KNOWN_HOSTS="$ROOT/40-deploy/rk3566/ssh/known_hosts"

exec ssh -F "$SSH_CONFIG" -o "UserKnownHostsFile=$KNOWN_HOSTS" \
    xrange-rk3566-1 /home/xduck1/xrange/60-tools/build-rk3566-microduck.sh
