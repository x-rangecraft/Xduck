#!/bin/sh
set -eu

TOOLS="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"

case "${1:-}" in
    check) "$TOOLS/sync-rk3566.sh" --check ;;
    sync) "$TOOLS/sync-rk3566.sh" --apply ;;
    build) "$TOOLS/remote-build-rk3566.sh" ;;
    deploy) "$TOOLS/deploy-rk3566.sh" ;;
    all)
        "$TOOLS/sync-rk3566.sh" --apply
        "$TOOLS/remote-build-rk3566.sh"
        ;;
    ssh) exec "$TOOLS/ssh-rk3566.sh" ;;
    *)
        echo "usage: $0 {check|sync|build|deploy|all|ssh}" >&2
        exit 2
        ;;
esac
