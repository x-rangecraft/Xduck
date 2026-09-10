#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
export JAVA_HOME="$ROOT/20-build/android/toolchain/jdk-17.0.20.1+1/Contents/Home"
export ANDROID_HOME="$ROOT/20-build/android/toolchain/sdk"
export ANDROID_USER_HOME="$ROOT/20-build/android/android-home"
export ANDROID_AVD_HOME="$ROOT/20-build/android/avd"
mkdir -p "$ANDROID_AVD_HOME" "$ANDROID_USER_HOME"
if [ ! -f "$ANDROID_AVD_HOME/duck-test.ini" ]; then
    printf 'no\n' | "$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager" create avd --name duck-test --package 'system-images;android-35;google_apis;arm64-v8a' --device pixel_5 --path "$ANDROID_AVD_HOME/duck-test.avd"
fi
exec "$ANDROID_HOME/emulator/emulator" -avd duck-test -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader_indirect -port 5580
