#!/usr/bin/env bash
set -euo pipefail

ROOT="${XRANGE_ROOT:-$HOME/xrange}"
SOURCE="$ROOT/10-source/rk3566/microduck"
BUILD="$ROOT/20-build/rk3566/microduck/release"
BUILD_TOOL="$ROOT/60-tools/build-rk3566-microduck.sh"
KEY_DIR="$ROOT/40-deploy/rk3566/secrets"
DEPLOY_ROOT="$ROOT/40-deploy/rk3566/microduck"
LOG_ROOT="$ROOT/50-logs/deploy/rk3566/microduck"
STAMP="$(date +%Y%m%d-%H%M%S)"
REVISION="${DUCK_REVISION:-working-tree}"
SHORT_REV="${REVISION:0:7}"
RELEASE_VERSION="0.10.0-dev.local.$(date +%s).g${SHORT_REV}"
WORK="$DEPLOY_ROOT/$RELEASE_VERSION"
STAGED="$WORK/staged"
DIST="$WORK/dist"
LOG="$LOG_ROOT/deploy-$STAMP.log"
XTASK="$BUILD/xtask"
UPDATER="$BUILD/updaterd"
KEY="$KEY_DIR/local-deploy.dev.key"
PUBLIC_KEY="$KEY_DIR/local-deploy.dev.pub"
CAMERA_RULE="$ROOT/40-deploy/rk3566/99-xrange-camera.rules"
ROBOTD_STOP_TIMEOUT="$ROOT/40-deploy/rk3566/10-robotd-stop-timeout.conf"
BOARD_BINS=(updaterd robotctl robotd configd btd padd mediad sounds pet-detect pet-features tofd xpad-usbd)

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG") 2>&1

[[ "$(uname -m)" == aarch64 ]] || { echo "error: expected aarch64" >&2; exit 1; }
. /etc/os-release
[[ "$ID" == debian && "$VERSION_ID" == 12 ]] || { echo "error: expected Debian 12" >&2; exit 1; }
[[ "$(hostname)" == rk3566-1 ]] || { echo "error: wrong host" >&2; exit 1; }
[[ "$(cat /etc/machine-id)" == 0160c5af701d40ad82a694adb73caf30 ]] \
    || { echo "error: wrong machine-id" >&2; exit 1; }
[[ "$(cat "$ROOT/.xrange-managed")" == xrange-managed-v1 ]] \
    || { echo "error: invalid xrange management marker" >&2; exit 1; }
command -v bwrap >/dev/null 2>&1 \
    || { echo "error: bubblewrap is required for isolated two-file policy workers" >&2; exit 1; }
command -v setpriv >/dev/null 2>&1 \
    || { echo "error: util-linux setpriv is required for isolated two-file policy workers" >&2; exit 1; }
if [[ -e /opt/robot/daemon/current ]]; then
    FIRST_INSTALL=false
else
    FIRST_INSTALL=true
fi

# A deployment must never package whatever happens to be left in `20-build` from an earlier
# source revision. Build and architecture-check the managed source first; the build tool also
# records its own log and immutable artifact copy in the prescribed partitions.
[[ -x "$BUILD_TOOL" ]] || { echo "error: missing build tool: $BUILD_TOOL" >&2; exit 1; }
"$BUILD_TOOL"

mkdir -p "$KEY_DIR" "$STAGED" "$DIST"
chmod 0700 "$KEY_DIR"

# xtask treats its current working directory as the repository boundary.  Run it
# from the Microduck source tree so the managed key directory remains outside
# the repository while still living under the xrange deployment partition.
cd "$SOURCE"
if [[ ! -f "$KEY" || ! -f "$PUBLIC_KEY" ]]; then
    "$XTASK" keygen --kind dev --name local-deploy --out "$KEY_DIR"
fi
chmod 0600 "$KEY"
chmod 0644 "$PUBLIC_KEY"
"$XTASK" keycheck --key "$KEY" --public "$PUBLIC_KEY"

for name in "${BOARD_BINS[@]}"; do
    [[ -x "$BUILD/$name" ]] || { echo "error: missing build binary: $name" >&2; exit 1; }
    file "$BUILD/$name" | grep -q 'ELF 64-bit.*ARM aarch64' \
        || { echo "error: $name is not ARM aarch64" >&2; exit 1; }
    cp -p "$BUILD/$name" "$STAGED/$name"
done

# This K11C target runs Debian 12. Do not package the upstream Debian 13
# GStreamer/RKAIQ provisioners: preinstall would run them on every release and
# could replace the board-validated Rockchip media stack with an incompatible ABI.
"$XTASK" package \
    --version "$RELEASE_VERSION" \
    --allow-version-drift \
    --channel daemon \
    --bin-dir "$STAGED" \
    --out "$DIST" \
    --revision "$REVISION" \
    --zstd-level 1 \
    --include "updater/systemd/updaterd.service=systemd/updaterd.service" \
    --include "updater/systemd/sysusers.d/robot.conf=systemd/sysusers.d/robot.conf" \
    --include "robotd/systemd/robotd.service=systemd/robotd.service" \
    --include "robotd/systemd/robot-policy-performance.service=systemd/robot-policy-performance.service" \
    --include "robotd/scripts/robot-policy-cpu-governor=scripts/robot-policy-cpu-governor" \
    --include "robotd/systemd/sysusers.d/robot-policy.conf=systemd/sysusers.d/robot-policy.conf" \
    --include "hooks/postinstall=hooks/postinstall" \
    --include "duck-detect/models/duck_detect.rknn=models/duck_detect.rknn" \
    --include "duck-detect/models/duck_detect.onnx=models/duck_detect.onnx" \
    --include "scripts/setup-npu.sh=scripts/setup-npu.sh" \
    --include "deploy/overlays/rk3568-npu-enable.dts=deploy/overlays/rk3568-npu-enable.dts" \
    --include "scripts/setup-login.sh=scripts/setup-login.sh" \
    --include "scripts/robot-rescue=scripts/robot-rescue" \
    --include "scripts/robot-boot-check=scripts/robot-boot-check" \
    --include "updater/systemd/robot-boot-check.service=systemd/robot-boot-check.service" \
    --include "updater/systemd/robot-boot-check.timer=systemd/robot-boot-check.timer" \
    --include "configd/systemd/configd.service=systemd/configd.service" \
    --include "btd/systemd/btd.service=systemd/btd.service" \
    --include "btd/systemd/sysusers.d/btd.conf=systemd/sysusers.d/btd.conf" \
    --include "padd/systemd/padd.service=systemd/padd.service" \
    --include "padd/systemd/sysusers.d/padd.conf=systemd/sysusers.d/padd.conf" \
    --include "drivers/flydigi-xpad/xpad-usbd.service=systemd/xpad-usbd.service" \
    --include "mediad/systemd/mediad.service=systemd/mediad.service" \
    --include "mediad/systemd/sysusers.d/mediad.conf=systemd/sysusers.d/mediad.conf" \
    --include "tof/systemd/tofd.service=systemd/tofd.service" \
    --include "tof/systemd/sysusers.d/tofd.conf=systemd/sysusers.d/tofd.conf" \
    --include "deploy/journald.conf.d/10-robot.conf=deploy/journald.conf.d/10-robot.conf" \
    --include "docs/design/architecture.md=docs/architecture.md" \
    --include "docs/design/updater-design.md=docs/updater-design.md" \
    --include "deploy/README.md=docs/deploy.md" \
    --include "policies/alpha_walking.onnx=policies/alpha_walking.onnx" \
    --include "policies/alpha_stand.onnx=policies/alpha_stand.onnx" \
    --include "policies/alpha_sitstand.onnx=policies/alpha_sitstand.onnx" \
    --include "policies/alpha_ground_pick.onnx=policies/alpha_ground_pick.onnx" \
    --include "policies/ball_kick_left.onnx=policies/ball_kick_left.onnx" \
    --include "policies/ball_kick_right.onnx=policies/ball_kick_right.onnx" \
    --include "policies/roller.onnx=policies/roller.onnx" \
    --include "policies/roller_crouch.onnx=policies/roller_crouch.onnx" \
    --include "policies/roulade.onnx=policies/roulade.onnx" \
    --include "pet-detect/models/pet_detect.onnx=models/pet_detect.onnx"
"$XTASK" sign --dir "$DIST" --key "$KEY"

sudo install -d -m 0755 /etc/robot /etc/robot/trusted_keys
if [[ ! -f /etc/robot/updater.toml ]]; then
    sudo install -m 0644 "$SOURCE/deploy/updater.toml" /etc/robot/updater.toml
    sudo sed -i 's/^allow_dev_keys[[:space:]]*=.*/allow_dev_keys        = true/' /etc/robot/updater.toml
    # This bench board has no release repository configured yet.  Do not poll the
    # literal ORG/duck-daemon placeholder or permit unattended installs.
    sudo sed -i \
        -e 's/^check_interval[[:space:]]*=/# check_interval =/' \
        -e 's/^auto_apply[[:space:]]*=.*/auto_apply = "off"/' \
        /etc/robot/updater.toml
fi
if [[ ! -f /etc/robot/robotd.toml ]]; then
    sudo install -m 0644 "$SOURCE/deploy/robotd.toml" /etc/robot/robotd.toml
fi
# This K11C uses the BL-1080P-S10 UVC camera (05a3:9230). Keep the USB transport
# compressed and decode it in MPP; its only native 16:9 30 fps mode is 1080p30.
sudo sed -i 's/^# camera = true$/camera = true/' /etc/robot/robotd.toml
sudo sed -i 's/^camera = false$/camera = true/' /etc/robot/robotd.toml
if sudo grep -q '^camera_backend[[:space:]]*=' /etc/robot/robotd.toml; then
    sudo sed -i 's/^camera_backend[[:space:]]*=.*/camera_backend = "uvc-mjpeg"/' /etc/robot/robotd.toml
else
    sudo sed -i '/^camera = true$/a camera_backend = "uvc-mjpeg"' /etc/robot/robotd.toml
fi
if sudo grep -q '^camera_device[[:space:]]*=' /etc/robot/robotd.toml; then
    sudo sed -i 's|^camera_device[[:space:]]*=.*|camera_device = "/dev/video-xrange-camera"|' /etc/robot/robotd.toml
else
    sudo sed -i '/^camera_backend = "uvc-mjpeg"$/a camera_device = "/dev/video-xrange-camera"' /etc/robot/robotd.toml
fi
sudo sed -i 's/^# camera_backend = "rkisp"$/camera_backend = "uvc-mjpeg"/' /etc/robot/robotd.toml
sudo sed -i 's|^# camera_device = "/dev/video0"$|camera_device = "/dev/video-xrange-camera"|' /etc/robot/robotd.toml
sudo sed -i 's/^# quality = "720p30"$/quality = "1080p30"/' /etc/robot/robotd.toml
sudo sed -i 's/^quality = "360p30"$/quality = "1080p30"/' /etc/robot/robotd.toml
sudo sed -i 's/^# rotation = 90$/rotation = 0/' /etc/robot/robotd.toml
if ! sudo grep -q '^rotation[[:space:]]*=' /etc/robot/robotd.toml; then
    sudo sed -i '/^quality = "1080p30"$/a rotation = 0' /etc/robot/robotd.toml
fi
# The vendor DAC mixer drops writes to zero. Keep its board gain and control all
# robot sounds through ALSA softvol. This file is owned by this board deployment.
sudo install -d -m 0755 /etc/alsa/conf.d
sudo tee /etc/alsa/conf.d/99-xduck-speaker.conf >/dev/null <<'ALSA'
pcm.xduck_speaker {
    type softvol
    slave.pcm "plughw:rockchiprk809"
    control {
        name "Xduck Playback Volume"
        card "rockchiprk809"
    }
    min_dB -51.0
    max_dB 0.0
    resolution 256
}
ALSA
# One silent open creates the mixer after boot; subsequent opens preserve its value.
sudo aplay -q -D xduck_speaker -t raw -f S16_LE -r 48000 -c 2 -d 1 /dev/zero
sudo amixer -M -c rockchiprk809 sget Xduck >/dev/null
sudo sed -i '/^\[audio\]/,/^\[/ { s/^\(# \)\?enabled = .*/enabled = true/; s/^\(# \)\?device = .*/device = "xduck_speaker"/; }' /etc/robot/robotd.toml
for release_key in "$SOURCE"/deploy/trusted_keys/release-*.pub; do
    sudo install -m 0644 "$release_key" "/etc/robot/trusted_keys/$(basename "$release_key")"
done
sudo install -m 0644 "$PUBLIC_KEY" /etc/robot/trusted_keys/local-deploy.dev.pub

# Device identity must exist before an update restarts mediad.
sudo install -m 0644 "$CAMERA_RULE" /etc/udev/rules.d/99-xrange-camera.rules
sudo install -d -m 0755 /etc/systemd/system/robotd.service.d
sudo install -m 0644 "$ROBOTD_STOP_TIMEOUT" \
    /etc/systemd/system/robotd.service.d/10-stop-timeout.conf
sudo systemctl daemon-reload
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=video4linux
sudo udevadm settle

if [[ "$FIRST_INSTALL" == true ]]; then
    echo "==> signed dry-run"
    sudo "$UPDATER" --config /etc/robot/updater.toml install --from "$DIST" --dry-run
    echo "==> first install"
    sudo "$UPDATER" --config /etc/robot/updater.toml install --from "$DIST"
else
    echo "==> signed update through updaterd"
    sudo /opt/robot/daemon/current/bin/robotctl update apply daemon --from "$DIST"
fi

sudo install -d -m 0755 /etc/systemd/journald.conf.d
sudo install -m 0644 /opt/robot/daemon/current/deploy/journald.conf.d/10-robot.conf \
    /etc/systemd/journald.conf.d/10-robot.conf
sudo ln -sfn /opt/robot/daemon/current/bin/robotctl /usr/local/bin/robotctl
sudo usermod -aG robot xduck1
sudo systemctl daemon-reload

if ! GST_PLUGIN_PATH=/usr/local/lib/gstreamer-1.0 gst-inspect-1.0 webrtcsink >/dev/null 2>&1 \
    || ! GST_PLUGIN_PATH=/usr/local/lib/gstreamer-1.0 gst-inspect-1.0 mpph264enc >/dev/null 2>&1 \
    || ! gst-inspect-1.0 nicesrc >/dev/null 2>&1 \
    || ! gst-inspect-1.0 intervideosrc >/dev/null 2>&1 \
    || ! gst-inspect-1.0 intervideosink >/dev/null 2>&1; then
    echo "error: mediad requires webrtcsink, mpph264enc, libnice and intervideo GStreamer plugins" >&2
    exit 1
fi

sudo systemctl disable --now xboxdrv.service >/dev/null 2>&1 || true
for unit in updaterd robotd configd btd xpad-usbd padd mediad; do
    sudo systemctl enable "$unit.service" >/dev/null 2>&1 || true
    sudo systemctl start "$unit.service" || true
done
# The ToF sensor is not attached on this bench assembly.  Keep the daemon in
# the release so it can be enabled later, but do not run an idle hardware poller.
sudo systemctl disable --now tofd.service >/dev/null 2>&1 || true
sudo systemctl enable robot-boot-check.timer >/dev/null 2>&1 || true

echo "version=$RELEASE_VERSION"
echo "log=$LOG"
echo "deploy_dir=$WORK"
