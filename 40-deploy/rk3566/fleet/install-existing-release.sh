#!/usr/bin/env bash
# Fresh same-hardware devices only. Reuse an archived, signed release; never build here.
# Run from the staged release directory, as root:
# bash install-existing-release.sh 122 SOC_SERIAL STM32_BY_ID NEW_MACHINE_ID EXPECTED_RELEASE
set -euo pipefail
DEVICE=${1:?device required}
SOC=${2:?SoC serial required}
STM32=${3:?STM32 by-id filename required}
NEW_ID=${4:?new machine-id required}
EXPECTED_RELEASE=${5:?expected release version required}
case "$DEVICE" in
    122) OPERATOR=xduck2; EXPECTED_HOST=rk3566-2 ;;
    124) OPERATOR=xduck3; EXPECTED_HOST=rk3566-3 ;;
    *) echo 'Only fresh 122 or 124 devices are supported' >&2; exit 2 ;;
esac
[[ $(id -u) == 0 && $(hostname) == "$EXPECTED_HOST" ]]
[[ $(tr -d '\0' </proc/device-tree/serial-number) == "$SOC" ]]
[[ $(uname -m) == aarch64 && $(uname -r) == 6.1.141 ]]
. /etc/os-release
[[ $ID == debian && $VERSION_ID == 12 ]]
tr '\0' '\n' </proc/device-tree/compatible | grep -qx rockchip,rk3566-kickpi-k11c
[[ $NEW_ID =~ ^[0-9a-f]{32}$ ]]
[[ $STM32 =~ ^usb-STMicroelectronics_STM32_Virtual_ComPort_[A-Fa-f0-9]+-if00$ ]]
[[ -e /dev/serial/by-id/$STM32 ]]
[[ ! -e /opt/robot/daemon/current && ! -L /opt/robot/daemon/current ]]
# A model runtime may be staged here; existing user data must not be replaced.
[[ ! -e /var/lib/robotd/policies && ! -e /var/lib/robotd/motor-calibration.json ]]
[[ ! -e /etc/robot/robotd.toml && ! -e /etc/robot/updater.toml ]]
STAGE=$PWD
ROOT=/home/$OPERATOR/xrange
LOG_DIR=$ROOT/50-logs/deploy/rk3566/fleet
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/install-$DEVICE-$(date +%Y%m%d-%H%M%S).log") 2>&1
sha256sum -c SHA256SUMS
VERSION=$(python3 -c 'import json; print(json.load(open("dist/manifest.json"))["version"])')
python3 - "$EXPECTED_RELEASE" <<'PY'
import hashlib, json, pathlib, sys
m = json.loads(pathlib.Path('dist/manifest.json').read_text())
assert m['version'] == sys.argv[1]
p = pathlib.Path('dist') / m['url']
assert p.parent == pathlib.Path('dist')
assert p.stat().st_size == m['size']
assert hashlib.sha256(p.read_bytes()).hexdigest() == m['sha256']
PY
for path in 00-docs 10-source 20-build 30-artifacts 40-deploy 50-logs 60-tools 90-temp; do
    install -d -o "$OPERATOR" -g "$OPERATOR" "$ROOT/$path"
done
install -d -m 0700 "$STAGE/device-backup"
cp -p /etc/machine-id "$STAGE/device-backup/machine-id.before"
printf '%s\n' "$NEW_ID" >/etc/machine-id
systemctl restart systemd-journald.service
journalctl --flush
printf 'device=%s\nhostname=%s\nsoc_serial=%s\nmachine_id=%s\nstm32=%s\nrelease=%s\n' \
    "$DEVICE" "$EXPECTED_HOST" "$SOC" "$NEW_ID" "$STM32" "$VERSION" >"$STAGE/device-identity.txt"
tar --no-same-owner -xzf runtime.tar.gz -C /
/sbin/ldconfig
for plugin in webrtcsink mpph264enc mppjpegdec nicesrc intervideosrc intervideosink; do
    GST_PLUGIN_PATH=/usr/local/lib/gstreamer-1.0 gst-inspect-1.0 "$plugin" >/dev/null
done
command -v bwrap
command -v setpriv
test -x /usr/sbin/irqbalance
/var/lib/robotd/model-python/bin/python -I -c 'import numpy, onnxruntime; print("policy runtime ready", numpy.__version__, onnxruntime.__version__)'

install -d -m 0755 /etc/robot /etc/robot/trusted_keys
install -m 0644 local-deploy.dev.pub /etc/robot/trusted_keys/local-deploy.dev.pub
for key in trusted_keys/release-*.pub; do
    install -m 0644 "$key" /etc/robot/trusted_keys/
done
install -m 0644 updater.toml /etc/robot/updater.toml
install -m 0644 robotd-baseline.toml /etc/robot/robotd.toml
python3 - "$STM32" <<'PY'
import pathlib, sys
p = pathlib.Path('/etc/robot/robotd.toml')
s = p.read_text()
old = 'usb-STMicroelectronics_STM32_Virtual_ComPort_385933373134-if00'
assert s.count(old) == 1
p.write_text(s.replace(old, sys.argv[1]))
PY
install -m 0644 99-xrange-camera.rules /etc/udev/rules.d/99-xrange-camera.rules
install -d -m 0755 /etc/alsa/conf.d
install -m 0644 99-xduck-speaker.conf /etc/alsa/conf.d/99-xduck-speaker.conf
aplay -q -D xduck_speaker -t raw -f S16_LE -r 48000 -c 2 -d 1 /dev/zero
install -d -m 0755 /etc/systemd/system/robotd.service.d /etc/systemd/system/irqbalance.service.d
install -m 0644 10-stop-timeout.conf /etc/systemd/system/robotd.service.d/10-stop-timeout.conf
install -m 0644 10-policy-cpu.conf /etc/systemd/system/irqbalance.service.d/10-policy-cpu.conf
udevadm control --reload-rules
udevadm trigger --subsystem-match=video4linux
udevadm settle
systemctl daemon-reload
systemctl restart irqbalance.service
# The updater's own engine verifies signatures before installation.
bootstrap/bin/updaterd --config /etc/robot/updater.toml install --from "$STAGE/dist" --dry-run
# Prevent postinstall's enable --now from polling unattached ToF hardware.
systemctl mask tofd.service
trap 'systemctl unmask tofd.service; systemctl disable --now tofd.service || true' EXIT
bootstrap/bin/updaterd --config /etc/robot/updater.toml install --from "$STAGE/dist"
systemctl unmask tofd.service
systemctl disable --now tofd.service
trap - EXIT
ln -sfn /opt/robot/daemon/current/bin/robotctl /usr/local/bin/robotctl
install -d -m 0755 /etc/systemd/journald.conf.d
install -m 0644 /opt/robot/daemon/current/deploy/journald.conf.d/10-robot.conf /etc/systemd/journald.conf.d/10-robot.conf
systemctl restart systemd-journald.service
journalctl --flush
usermod -aG robot "$OPERATOR"
systemctl disable --now xboxdrv.service 2>/dev/null || true
systemctl daemon-reload
for unit in updaterd robotd configd btd xpad-usbd padd mediad; do
    systemctl enable --now "$unit.service"
done
systemctl enable robot-boot-check.timer
test "$(readlink -f /opt/robot/daemon/current)" = "/opt/robot/daemon/releases/$VERSION"
cmp bootstrap/bin/updaterd /opt/robot/daemon/current/bin/updaterd
robotctl health
printf 'Installed baseline %s on device %s; motor calibration and hardware acceptance remain separate.\n' "$VERSION" "$DEVICE"
