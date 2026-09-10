#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
TASK_JDK="${JAVA_HOME:-$ROOT/20-build/android/toolchain/jdk-17.0.20.1+1/Contents/Home}"
OUT="$ROOT/20-build/android/wire-checks"
SRC="$ROOT/10-source/android/duck-ble/app/src"
mkdir -p "$OUT"
"$TASK_JDK/bin/javac" -d "$OUT" "$SRC/main/java/com/microduck/ble/Wire.java" "$SRC/main/java/com/microduck/ble/ConnectionLifecycle.java" "$SRC/testHarness/java/com/microduck/ble/WireChecks.java" "$SRC/testHarness/java/com/microduck/ble/ConnectionLifecycleChecks.java"
"$TASK_JDK/bin/java" -cp "$OUT" com.microduck.ble.WireChecks
"$TASK_JDK/bin/java" -cp "$OUT" com.microduck.ble.ConnectionLifecycleChecks
