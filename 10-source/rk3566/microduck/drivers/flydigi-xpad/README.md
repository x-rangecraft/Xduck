# Flydigi Xbox-compatible USB driver

This is the hardware-adaptation layer for the Flydigi Dune Fox receiver used on
the RK3566 K11C assembly. The receiver enumerates as the Xbox 360-compatible USB
ID `045e:028e`, but Debian's `xboxdrv 0.8.8` does not publish input events for its
reports. The vendor kernel also has no loadable `xpad` module.

`xpad-usbd` claims USB interface 0, reads endpoint `0x81`, decodes the standard
20-byte Xbox 360 input report, and publishes a standard Linux evdev controller
through `/dev/uinput`. It owns no robot behavior: `padd` remains the only layer
that maps evdev controls to robot intents.

The daemon survives controller/receiver removal. It destroys the virtual evdev
device immediately, retries discovery at a low rate, and recreates the device
after reconnection. Runtime logs belong to journald under `xpad-usbd.service`.

Build and parser check:

```sh
make -C drivers/flydigi-xpad clean all test
```
