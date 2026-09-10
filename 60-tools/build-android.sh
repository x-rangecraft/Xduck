#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
TOOLCHAIN="$ROOT/20-build/android/toolchain"
export JAVA_HOME="${JAVA_HOME:-$TOOLCHAIN/jdk-17.0.20.1+1/Contents/Home}"
export ANDROID_HOME="${ANDROID_HOME:-$TOOLCHAIN/sdk}"
export GRADLE_USER_HOME="$ROOT/20-build/android/gradle-home"
export ANDROID_USER_HOME="$ROOT/20-build/android/android-home"
export PATH="$JAVA_HOME/bin:$PATH"
GRADLE="${GRADLE_BIN:-$TOOLCHAIN/gradle-8.11.1/bin/gradle}"
mkdir -p "$GRADLE_USER_HOME" "$ANDROID_USER_HOME" "$ROOT/30-artifacts/android" "$ROOT/50-logs/build/android"
cd "$ROOT/10-source/android/duck-ble"
"$GRADLE" --project-cache-dir "$ROOT/20-build/android/project-cache" :app:assembleDebug :app:lintDebug
cp "$ROOT/20-build/android/duck-ble/app/outputs/apk/debug/app-debug.apk" "$ROOT/30-artifacts/android/Xduck-BLE-0.1.3-debug.apk"
cd "$ROOT/30-artifacts/android"
shasum -a 256 Xduck-BLE-0.1.3-debug.apk > SHA256SUMS
