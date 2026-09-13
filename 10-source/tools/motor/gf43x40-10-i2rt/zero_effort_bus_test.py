"""Bounded 300-second, all-14-motor zero-effort DMUSB commissioning test.

Requires robotd stopped. Uses existing firmware gates, temporary measured-position
limits, all explicit IDs, and verified disable in finally. Never zeros encoders.
"""
import json
import os
import signal
import struct
import subprocess
import time
from probe_stm32_fdcan import Link, state

COUNT = 14
MASK = (1 << COUNT) - 1
DURATION = 300.0


def emit(event, **data):
    print(json.dumps(dict(event=event, **data)), flush=True)


def admin(link, op):
    return link.send(12, struct.pack('<HBB24B', 0, op, COUNT,
                                    *range(1, COUNT+1), *([0]*(24-COUNT))))


def ack(link, seq, op):
    end = time.monotonic()+2
    while time.monotonic() < end:
        for kind, p in link.frames():
            if kind == 13 and struct.unpack_from('<H', p)[0] == seq:
                emit('ack', op=op, raw=p.hex())
                if (len(p) != 28 or p[2] != op or p[3] or p[20]
                        or p[21:23] != bytes([COUNT, COUNT])
                        or struct.unpack_from('<II', p, 12) != (MASK, MASK)
                        or (op == 1 and struct.unpack_from('<I', p, 8)[0] != 0)):
                    raise RuntimeError('firmware rejected operation')
                return struct.unpack_from('<I', p, 4)[0]
    raise TimeoutError('admin acknowledgement')


def command(positions):
    return (struct.pack('<HBBI', 0, 1, COUNT, 0)
            + b''.join(struct.pack('<iiiHHHH', round(p*1000), 0, 0, 0, 0, 1, 0)
                       for p in positions))


def limits_payload(positions):
    return (struct.pack('<HBB', 0, COUNT, 0)
            + b''.join(struct.pack('<B3xii', i+1, round((p-.15)*1000), round((p+.15)*1000))
                       for i, p in enumerate(positions)))


def check(s, enabled, positions=None):
    if s['count'] != COUNT or s['faults'] or s['imu_flags'] & 3 != 3:
        raise RuntimeError('gateway, IMU, or motor count failed')
    for i, m in enumerate(s['motors']):
        if m['flags'] & 15 != (7 if enabled else 3) or m['age_ms'] > 100 or m['temp'] >= 60:
            raise RuntimeError(f'motor {i+1} feedback/temperature failed: {m}')
        if abs(m['v']) > (.5 if enabled else .1):
            raise RuntimeError(f'motor {i+1} speed exceeded test bound')
        if positions is not None and abs(m['p']-positions[i]) > .05:
            raise RuntimeError(f'motor {i+1} moved beyond 0.05 rad')


def main():
    if subprocess.run(['systemctl', 'is-active', '--quiet', 'robotd.service']).returncode == 0:
        raise SystemExit('robotd must be stopped')
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    link = Link('/dev/ttyACM0')
    attempted = False
    try:
        samples = []
        end = time.monotonic()+2
        while time.monotonic() < end:
            for kind, p in link.frames():
                if kind == 2: samples.append(state(p))
        if len(samples) < 50: raise RuntimeError('insufficient baseline states')
        for s in samples: check(s, False)
        positions = [m['p'] for m in samples[-1]['motors']]
        if any(abs(p) > 3 for p in positions): raise RuntimeError('unexpected starting position')
        for s in samples: check(s, False, positions)
        emit('baseline', pid=os.getpid(), frames=len(samples), state=samples[-1])
        limits = limits_payload(positions)
        ack(link, link.send(14, limits), 4)
        attempted = True
        ack(link, admin(link, 1), 1)
        cmd = command(positions)
        start = time.monotonic()
        last_state = start
        last_tx = 0.0
        next_report = start+30
        count = tx = 0
        max_age = [0]*COUNT
        max_move = [0.0]*COUNT
        max_temp = [0]*COUNT
        max_speed = [0.0]*COUNT
        max_gap = max_tx_gap = 0.0
        last_tick = None
        while time.monotonic()-start < DURATION:
            now = time.monotonic()
            if now-last_state > .1: raise RuntimeError('USB state stream stale')
            if now-last_tx >= .02:
                if tx: max_tx_gap = max(max_tx_gap, now-last_tx)
                link.send(1, cmd)
                tx += 1
                last_tx = now
            for kind, p in link.frames():
                if kind != 2: continue
                s = state(p)
                # Discard only states produced before successful ENABLE acknowledgement.
                if s['tick'] < samples[-1]['tick']: raise RuntimeError('STM32 restarted')
                if count == 0 and any(m['flags'] & 15 != 7 for m in s['motors']):
                    continue
                check(s, True, positions)
                if last_tick is not None:
                    gap = (s['tick']-last_tick) & 0xffffffff
                    if gap == 0 or gap > 100: raise RuntimeError('state timestamp gap')
                    max_gap = max(max_gap, gap)
                last_tick = s['tick']
                last_state = time.monotonic()
                count += 1
                for i, m in enumerate(s['motors']):
                    max_age[i] = max(max_age[i], m['age_ms'])
                    max_move[i] = max(max_move[i], abs(m['p']-positions[i]))
                    max_temp[i] = max(max_temp[i], m['temp'])
                    max_speed[i] = max(max_speed[i], abs(m['v']))
            if now >= next_report:
                emit('progress', elapsed_s=now-start, frames=count, commands=tx,
                     max_age_ms=max_age, max_move_rad=max_move, max_temp_c=max_temp,
                     max_speed_rad_s=max_speed, max_state_gap_ms=max_gap,
                     max_host_command_gap_ms=max_tx_gap*1000)
                next_report += 30
        if count < 14000: raise RuntimeError('too few valid enabled samples')
        emit('completed', elapsed_s=time.monotonic()-start, frames=count, commands=tx,
             max_age_ms=max_age, max_move_rad=max_move, max_temp_c=max_temp,
             max_speed_rad_s=max_speed, max_state_gap_ms=max_gap,
             max_host_command_gap_ms=max_tx_gap*1000)
    finally:
        try:
            if attempted:
                at = ack(link, admin(link, 2), 2)
                end = time.monotonic()+2
                verified = None
                while time.monotonic() < end:
                    for kind, p in link.frames():
                        if kind != 2: continue
                        s = state(p)
                        if all(0 < ((s['tick']-m['age_ms']-at) & 0xffffffff) < 3000
                               and m['flags'] & 15 == 3 and m['age_ms'] < 100
                               for m in s['motors']):
                            verified = s
                            break
                    if verified is not None: break
                if verified is None: raise RuntimeError('fresh all-motor disable feedback missing')
                # SET_LIMITS succeeds only when every raw motor state is actually 0.
                ack(link, link.send(14, limits), 4)
                emit('disabled_verified', state=verified)
        finally:
            link.close()


if __name__ == '__main__':
    main()
