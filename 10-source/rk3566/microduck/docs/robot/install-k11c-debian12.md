# Install on a KICKPI K11C (Debian 12)

This is the supported path for the KICKPI K11C RK3566 board. Flash the **K11C-specific Debian 12
/ Linux 6.1** image from KICKPI; an image for another RK3566 board is not interchangeable because
U-Boot, DDR setup and the device tree are board-specific.

## Verify the image

On the board:

```sh
uname -m
uname -r
. /etc/os-release && printf '%s %s\n' "$ID" "$VERSION_ID"
tr '\0' '\n' </proc/device-tree/compatible
```

The first and third commands must report `aarch64` and `debian 12`. The compatible strings should
identify K11C. Stop if they do not: `setup-board.sh` deliberately refuses a K11C profile on another
distribution.

## Provision from this checkout

The K11C path keeps the vendor kernel and DTB. It does not install Radxa kernel packages or Radxa
overlays.

```sh
cd Rk3566
./scripts/provision-board.sh kickpi@BOARD_IP --board-profile kickpi-k11c --ref YOUR_BRANCH
```

Push the branch and let its build finish first: provisioning fetches the board-side scripts and
release artifacts from that ref. Omit `--ref` after these changes are on the repository default
branch.

The STM32 H7 motor/IMU gateway is expected at `/dev/ttyACM0`. Confirm it before expecting the
control health check to pass:

```sh
ls -l /dev/ttyACM0
dmesg | grep -i ttyACM
```

Override a genuinely different stable device path with `DUCK_MOTOR_PORT`; do not point it at the
K11C debug console.

## Current media limitation

The Radxa camera, TLV320AIC3104/ToF overlays, RKAIQ package and hardware GStreamer plugin bundle are
not installed on K11C. They were built and measured against the Radxa Zero 3W BSP. Provisioning
therefore forces those two Radxa installers off for K11C while installing the control, Bluetooth,
configuration and update services normally.

The NPU setup remains usable when the K11C vendor kernel already exposes the RKNPU driver: it
installs `librknnrt.so` without editing the board's boot configuration. Camera/audio acceleration
needs a separate K11C-specific bring-up after the actual sensor and codec wiring is known.
