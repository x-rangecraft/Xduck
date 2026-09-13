"""Observe DMUSB, or commission ONLY the explicit one-route ID1 bench image.

Run on the RK3566 with robotd stopped. No zeroing, home motion, or nonzero MIT
gains/effort. The normal 14-route firmware is rejected by --enable-check.
"""
import argparse
import binascii
import fcntl
import json
import os
import select
import struct
import subprocess
import termios
import time
import tty


def pack(kind, seq, payload):
    frame = bytearray(struct.pack('<HBBHHIIH', 0x4d47, 4, kind, seq,
                                  len(payload), 0, 0, 0) + payload)
    struct.pack_into('<H', frame, 16, binascii.crc_hqx(frame, 0xffff))
    return frame


class Link:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        fcntl.ioctl(self.fd, termios.TIOCEXCL)
        tty.setraw(self.fd)
        attrs = termios.tcgetattr(self.fd)
        attrs[4] = attrs[5] = getattr(termios, 'B1000000', termios.B115200)
        attrs[2] |= termios.CLOCAL | termios.CREAD
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        self.buffer = bytearray()
        self.seq = 0

    def send(self, kind, payload):
        self.seq = (self.seq + 1) & 0xffff
        payload = bytearray(payload)
        struct.pack_into('<H', payload, 0, self.seq)
        data = pack(kind, self.seq, payload)
        deadline = time.monotonic() + .05
        while data:
            if time.monotonic() > deadline:
                raise TimeoutError('USB write deadline')
            if select.select([], [self.fd], [], .005)[1]:
                data = data[os.write(self.fd, data):]
        return self.seq

    def frames(self):
        if select.select([self.fd], [], [], .01)[0]:
            self.buffer.extend(os.read(self.fd, 4096))
        result = []
        while len(self.buffer) >= 18:
            if self.buffer[:3] != b'GM\x04':
                del self.buffer[0]
                continue
            length = struct.unpack_from('<H', self.buffer, 6)[0]
            if length > 1536:
                del self.buffer[0]
                continue
            if len(self.buffer) < 18 + length:
                break
            frame = bytearray(self.buffer[:18 + length])
            expected = struct.unpack_from('<H', frame, 16)[0]
            frame[16:18] = b'\0\0'
            if binascii.crc_hqx(frame, 0xffff) != expected:
                del self.buffer[0]
                continue
            del self.buffer[:18 + length]
            result.append((frame[3], frame[18:]))
        return result

    def admin(self, op):
        return self.send(12, struct.pack('<HBB24B', 0, op, 1, 1, *([0]*23)))

    def close(self):
        fcntl.ioctl(self.fd, termios.TIOCNXCL)
        os.close(self.fd)


def state(payload):
    if len(payload) < 64 or len(payload) != 64 + 20 * payload[13]:
        raise ValueError('invalid state length')
    tick, faults = struct.unpack_from('<II', payload, 4)
    motors = []
    for slot in range(payload[13]):
        p, v, tau, rx, flags, temp, _ = struct.unpack_from('<iiiIBBH', payload, 64+20*slot)
        motors.append(dict(slot=slot, p=p/1000, v=v/1000, tau=tau/1000,
                           age_ms=(tick-rx) & 0xffffffff, flags=flags, temp=temp))
    return dict(tick=tick, faults=faults, count=payload[13],
                capabilities=struct.unpack_from('<H', payload, 14)[0],
                imu_flags=payload[60], motors=motors)


def wait_ack(link, seq, op):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        for kind, payload in link.frames():
            if kind == 13 and struct.unpack_from('<H', payload)[0] == seq:
                print(json.dumps(dict(event='ack', op=op, raw=payload.hex())), flush=True)
                if payload[2] != op or payload[3] != 0 or payload[20] or payload[21:23] != b'\1\1':
                    raise RuntimeError('gateway rejected operation')
                return struct.unpack_from('<I', payload, 4)[0]
    raise TimeoutError('administration acknowledgement')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--enable-check', action='store_true')
    args = ap.parse_args()
    if subprocess.run(['systemctl', 'is-active', '--quiet', 'robotd.service']).returncode == 0:
        raise SystemExit('Stop robotd before accessing its USB device')
    link = Link(args.port)
    attempted = False
    try:
        samples = []
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            for kind, payload in link.frames():
                if kind == 2:
                    samples.append(state(payload))
        if not samples:
            raise RuntimeError('no CRC-valid STM32 state frames')
        s = samples[-1]
        print(json.dumps(dict(event='baseline', frames=len(samples), state=s)), flush=True)
        if not args.enable_check:
            return
        if s['count'] != 1:
            raise RuntimeError('enable-check requires the isolated ID1 bench image')
        m = s['motors'][0]
        if s['faults'] or s['capabilities'] & 7 != 7 or s['imu_flags'] & 3 != 3:
            raise RuntimeError('gateway/IMU is not ready')
        if m['flags'] != 3 or m['age_ms'] > 100 or abs(m['p']) >= 3 or abs(m['v']) > .1 or m['temp'] >= 60:
            raise RuntimeError('ID1 lacks fresh stationary disabled feedback')
        limits = struct.pack('<HBB', 0, 1, 0) + struct.pack('<B3xii', 1, -3000, 3000)
        wait_ack(link, link.send(14, limits), 4)
        attempted = True
        wait_ack(link, link.admin(1), 1)
        deadline = time.monotonic() + .5
        last_tx = 0
        enabled_samples = []
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now-last_tx >= .02:
                # Keep the measured position field; Kp, Kd, v and tau are all zero.
                command = struct.pack('<HBBIiiiHHHH', 0, 1, 1, 0,
                                      round(m['p']*1000), 0, 0, 0, 0, 1, 0)
                link.send(1, command)
                last_tx = now
            for kind, payload in link.frames():
                if kind != 2:
                    continue
                current = state(payload)
                motor = current['motors'][0]
                if current['faults'] or motor['flags'] & 15 != 7 or motor['age_ms'] > 100:
                    raise RuntimeError('enabled feedback lost or protection triggered')
                if abs(motor['p']-m['p']) > .05 or abs(motor['v']) > .5:
                    raise RuntimeError('unexpected movement during zero-effort handshake')
                enabled_samples.append(current)
        if len(enabled_samples) < 5:
            raise RuntimeError('insufficient fresh enabled samples')
        print(json.dumps(dict(event='enable_verified', samples=len(enabled_samples),
                              first=enabled_samples[0], last=enabled_samples[-1])), flush=True)
    finally:
        try:
            if attempted:
                disabled_at = wait_ack(link, link.admin(2), 2)
                deadline = time.monotonic() + 1
                disabled = None
                while time.monotonic() < deadline:
                    for kind, payload in link.frames():
                        if kind == 2:
                            s = state(payload)
                            if (s['count'] == 1 and s['motors'][0]['flags'] & 15 == 3
                                    and s['motors'][0]['age_ms'] < 100
                                    and 0 < ((s['tick']-s['motors'][0]['age_ms']-disabled_at) & 0xffffffff) < 1000):
                                disabled = s
                if disabled is None:
                    raise RuntimeError('fresh post-disable feedback not confirmed')
                # USB flags alone cannot prove physical disable: require the
                # firmware's calibration gate, which checks actual state == 0.
                wait_ack(link, link.send(14, limits), 4)
                print(json.dumps(dict(event='disabled_after_test', state=disabled)), flush=True)
        finally:
            link.close()


if __name__ == '__main__':
    main()
