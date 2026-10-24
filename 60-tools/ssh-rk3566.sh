#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
SSH_CONFIG="$ROOT/40-deploy/rk3566/ssh/config"
KNOWN_HOSTS="$ROOT/40-deploy/rk3566/ssh/known_hosts"
TARGET=xrange-rk3566-1
if [ "${1:-}" = --device ]; then
    case "${2:-}" in
        121) TARGET=xrange-rk3566-1 ;;
        122) TARGET=xrange-rk3566-2 ;;
        124) TARGET=xrange-rk3566-3 ;;
        *) echo 'usage: ssh-rk3566.sh [--device 121|122|124] [ssh arguments]' >&2; exit 2 ;;
    esac
    shift 2
fi

exec ssh -F "$SSH_CONFIG" -o "UserKnownHostsFile=$KNOWN_HOSTS" "$TARGET" "$@"
