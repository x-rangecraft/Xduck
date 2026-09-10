#!/usr/bin/env bash
set -euo pipefail

ROOT="${XRANGE_ROOT:-$HOME/xrange}"
PROJECT="$ROOT/10-source/rk3566/microduck"
BUILD_ROOT="$ROOT/20-build/rk3566/microduck"
LOG_ROOT="$ROOT/50-logs/build/rk3566/microduck"
ARTIFACT_ROOT="$ROOT/30-artifacts/rk3566/microduck"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_ROOT/release-$STAMP.log"
ARTIFACT_DIR="$ARTIFACT_ROOT/native-release-$STAMP"
BOARD_BINS=(btd configd mediad padd robotctl robotd sounds tofd updaterd xpad-usbd)

[[ -f "$PROJECT/Cargo.toml" ]] || { echo "error: missing workspace: $PROJECT" >&2; exit 1; }
[[ "$(uname -m)" == aarch64 ]] || { echo "error: this builder is not aarch64" >&2; exit 1; }
. /etc/os-release
[[ "$ID" == debian && "$VERSION_ID" == 12 ]] || { echo "error: expected Debian 12" >&2; exit 1; }

mkdir -p "$BUILD_ROOT" "$LOG_ROOT" "$ARTIFACT_ROOT"
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="$BUILD_ROOT"

echo "source:   $PROJECT"
echo "build:    $BUILD_ROOT"
echo "log:      $LOG_FILE"
echo "artifact: $ARTIFACT_DIR"

cd "$PROJECT"
cargo build --locked --release --bins 2>&1 | tee "$LOG_FILE"

# The Flydigi receiver adaptation is intentionally below `padd`: a small USB/uinput
# driver turns the receiver's Xbox-compatible reports into standard evdev events.
# Build it separately because it is a Linux hardware driver, not a Rust robot daemon.
make -C drivers/flydigi-xpad clean all test 2>&1 | tee -a "$LOG_FILE"
install -m 0755 drivers/flydigi-xpad/xpad-usbd "$BUILD_ROOT/release/xpad-usbd"

mkdir -p "$ARTIFACT_DIR/bin"
for name in "${BOARD_BINS[@]}"; do
    binary="$BUILD_ROOT/release/$name"
    [[ -x "$binary" ]] || { echo "error: missing binary: $name" >&2; exit 1; }
    file "$binary" | grep -q 'ELF 64-bit.*ARM aarch64' \
        || { echo "error: $name is not an ARM aarch64 ELF" >&2; exit 1; }
    cp -p "$binary" "$ARTIFACT_DIR/bin/$name"
done

(
    cd "$ARTIFACT_DIR/bin"
    sha256sum "${BOARD_BINS[@]}" > ../SHA256SUMS
)

{
    echo "project=microduck"
    echo "architecture=rk3566/aarch64"
    echo "os=debian-12"
    echo "profile=release"
    echo "source=$PROJECT"
    echo "build_log=$LOG_FILE"
    rustc --version
    uname -a
} > "$ARTIFACT_DIR/BUILD-INFO.txt"

echo "build verified: $ARTIFACT_DIR"
