#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
failed=0

require_dir() {
    if [ ! -d "$ROOT/$1" ]; then
        echo "missing directory: $1" >&2
        failed=1
    fi
}

for dir in 00-docs 10-source 20-build 30-artifacts 40-deploy 50-logs 60-tools 90-temp; do
    require_dir "$dir"
done

for dir in 50-logs/build 50-logs/test 50-logs/deploy; do
    require_dir "$dir"
done

if [ -e "$ROOT/50-logs/runtime" ]; then
    echo "legacy log path must not exist: 50-logs/runtime" >&2
    failed=1
fi

for legacy in Rk3566 Stm32 Motor; do
    if [ -e "$ROOT/$legacy" ]; then
        echo "legacy path must not exist: $legacy" >&2
        failed=1
    fi
done

test -f "$ROOT/10-source/rk3566/microduck/Cargo.toml" \
    || { echo "missing RK3566 Cargo workspace" >&2; failed=1; }
git -C "$ROOT/10-source/rk3566/microduck" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || { echo "missing project Git metadata" >&2; failed=1; }
test -L "$ROOT/10-source/tools/motor/gf43x40-10-i2rt/data" \
    || { echo "motor data must be a link into 50-logs/test" >&2; failed=1; }

MANIFEST="$ROOT/40-deploy/rk3566/CONTROL_FILES.sha256"
if [ ! -f "$MANIFEST" ]; then
    echo "missing RK3566 control-file manifest" >&2
    failed=1
elif ! (cd "$ROOT" && shasum -a 256 -c "40-deploy/rk3566/CONTROL_FILES.sha256" >/dev/null); then
    echo "RK3566 control file changed; review it and update the manifest deliberately" >&2
    failed=1
fi

for executable in \
    60-tools/ssh-rk3566.sh \
    60-tools/sync-rk3566.sh \
    60-tools/remote-build-rk3566.sh \
    60-tools/deploy-rk3566.sh \
    60-tools/rk3566-workflow.sh \
    40-deploy/rk3566/remote/build-rk3566-microduck.sh \
    40-deploy/rk3566/remote/deploy-rk3566-microduck.sh; do
    test -x "$ROOT/$executable" \
        || { echo "control script is not executable: $executable" >&2; failed=1; }
done

test -f "$ROOT/10-source/rk3566/microduck/drivers/flydigi-xpad/xpad-usbd.c" \
    || { echo "missing K11C Flydigi USB driver source" >&2; failed=1; }
test -f "$ROOT/10-source/rk3566/microduck/drivers/flydigi-xpad/xpad-usbd.service" \
    || { echo "missing K11C Flydigi USB driver unit" >&2; failed=1; }
test -f "$ROOT/40-deploy/rk3566/remote/99-xrange-camera.rules" \
    || { echo "missing BL-1080P-S10 udev rule" >&2; failed=1; }

# Android BLE is a separate native app; generated outputs stay outside source.
for file in 10-source/android/duck-ble/settings.gradle 10-source/android/duck-ble/app/src/main/AndroidManifest.xml; do
    test -f "$ROOT/$file" || { echo "missing Android BLE source: $file" >&2; failed=1; }
done
for generated in 10-source/android/duck-ble/build 10-source/android/duck-ble/app/build; do
    test ! -d "$ROOT/$generated" || { echo "Android output must be in 20-build: $generated" >&2; failed=1; }
done

if [ "$failed" -ne 0 ]; then
    exit 1
fi

echo "layout ok: $ROOT"
