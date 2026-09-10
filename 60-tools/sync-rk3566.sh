#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
SOURCE="$ROOT/10-source/rk3566/microduck"
PROTOCOL="$ROOT/10-source/rk3566/STM32_DMUSB_V4_PROTOCOL.md"
DOC="$ROOT/00-docs/ENGINEERING_WORKFLOW.md"
REMOTE_BUILD_TOOL="$ROOT/40-deploy/rk3566/remote/build-rk3566-microduck.sh"
REMOTE_DEPLOY_TOOL="$ROOT/40-deploy/rk3566/remote/deploy-rk3566-microduck.sh"
REMOTE_README="$ROOT/40-deploy/rk3566/remote/README.md"
REMOTE_CAMERA_RULE="$ROOT/40-deploy/rk3566/remote/99-xrange-camera.rules"
REMOTE_ROBOTD_STOP_TIMEOUT="$ROOT/40-deploy/rk3566/remote/10-robotd-stop-timeout.conf"
SSH_CONFIG="$ROOT/40-deploy/rk3566/ssh/config"
KNOWN_HOSTS="$ROOT/40-deploy/rk3566/ssh/known_hosts"
HOST=xrange-rk3566-1
REMOTE_ROOT=/home/xduck1/xrange
REMOTE_SOURCE="$REMOTE_ROOT/10-source/rk3566/microduck"
REMOTE_PROTOCOL="$REMOTE_ROOT/10-source/rk3566/STM32_DMUSB_V4_PROTOCOL.md"
EXPECTED_ID='rk3566-1|0160c5af701d40ad82a694adb73caf30|aarch64|debian|12'
MODE="${1:---check}"

case "$MODE" in
    --check|--apply) ;;
    *) echo "usage: $0 [--check|--apply]" >&2; exit 2 ;;
esac

test -f "$SOURCE/Cargo.toml" \
    || { echo "error: missing local source: $SOURCE" >&2; exit 1; }
test -f "$PROTOCOL" \
    || { echo "error: missing protocol fact source: $PROTOCOL" >&2; exit 1; }

ssh_project() {
    ssh -F "$SSH_CONFIG" -o "UserKnownHostsFile=$KNOWN_HOSTS" "$HOST" "$@"
}

actual="$(ssh_project 'set -eu; . /etc/os-release; printf "%s|%s|%s|%s|%s" "$(hostname)" "$(cat /etc/machine-id)" "$(uname -m)" "$ID" "$VERSION_ID"')"
[ "$actual" = "$EXPECTED_ID" ] || {
    echo "error: target identity mismatch" >&2
    echo "expected: $EXPECTED_ID" >&2
    echo "actual:   $actual" >&2
    exit 1
}

marker="$(ssh_project "cat '$REMOTE_ROOT/.xrange-managed' 2>/dev/null || true")"
[ "$marker" = xrange-managed-v1 ] || {
    echo "error: missing or invalid remote management marker" >&2
    exit 1
}

RSYNC_RSH="ssh -F $SSH_CONFIG -o UserKnownHostsFile=$KNOWN_HOSTS"
export RSYNC_RSH
set -- -az --no-owner --no-group --delete-delay --itemize-changes \
    --exclude=.git/ --exclude=target/ --exclude=.DS_Store
[ "$MODE" = --apply ] || set -- "$@" --dry-run

echo "target identity: $actual"
echo "mode: $MODE"
rsync "$@" "$SOURCE/" "$HOST:$REMOTE_SOURCE/"

if [ "$MODE" = --apply ]; then
    rsync -az --no-owner --no-group "$PROTOCOL" "$HOST:$REMOTE_PROTOCOL"
    rsync -az --no-owner --no-group "$DOC" "$HOST:$REMOTE_ROOT/00-docs/ENGINEERING_WORKFLOW.md"
    rsync -az --no-owner --no-group "$REMOTE_README" "$HOST:$REMOTE_ROOT/README.md"
    rsync -az --no-owner --no-group "$REMOTE_BUILD_TOOL" "$HOST:$REMOTE_ROOT/60-tools/build-rk3566-microduck.sh"
    rsync -az --no-owner --no-group "$REMOTE_DEPLOY_TOOL" "$HOST:$REMOTE_ROOT/60-tools/deploy-rk3566-microduck.sh"
    rsync -az --no-owner --no-group "$REMOTE_CAMERA_RULE" "$HOST:$REMOTE_ROOT/40-deploy/rk3566/99-xrange-camera.rules"
    rsync -az --no-owner --no-group "$REMOTE_ROBOTD_STOP_TIMEOUT" "$HOST:$REMOTE_ROOT/40-deploy/rk3566/10-robotd-stop-timeout.conf"
    ssh_project "chgrp -R xduck1 '$REMOTE_SOURCE'; chgrp xduck1 '$REMOTE_PROTOCOL' '$REMOTE_ROOT/README.md' '$REMOTE_ROOT/00-docs/ENGINEERING_WORKFLOW.md' '$REMOTE_ROOT/60-tools/build-rk3566-microduck.sh' '$REMOTE_ROOT/60-tools/deploy-rk3566-microduck.sh' '$REMOTE_ROOT/40-deploy/rk3566/99-xrange-camera.rules' '$REMOTE_ROOT/40-deploy/rk3566/10-robotd-stop-timeout.conf'; chmod 0755 '$REMOTE_ROOT/60-tools/build-rk3566-microduck.sh' '$REMOTE_ROOT/60-tools/deploy-rk3566-microduck.sh'"
    echo "sync complete: $REMOTE_SOURCE"
else
    echo "dry run only; use --apply after reviewing the list"
fi
