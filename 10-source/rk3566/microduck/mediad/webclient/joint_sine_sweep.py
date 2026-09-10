#!/usr/bin/env python3
"""Offline sine-sweep file generator for uploaded tasks. Never connects or enables."""
import argparse
import json
import math
from pathlib import Path
from motor_experiment import write_task
AMPLITUDES = [0.05, 0.10, 0.15, 0.20]
FREQUENCIES = [0.2, 0.5, 1.0, 1.5, 2.0]

def trajectory(t, amplitude, frequency, ramp, hold):
    """C2 envelope; return position offset, velocity and acceleration (SI)."""
    duration = 2 * ramp + hold
    if t <= 0 or t >= duration:
        return 0.0, 0.0, 0.0
    if t < ramp:
        u, sign = t / ramp, 1
    elif t > ramp + hold:
        u, sign = (duration - t) / ramp, -1
    else:
        u, sign = 1.0, 0
    envelope = 10*u**3 - 15*u**4 + 6*u**5
    ed = sign * (30*u**2 - 60*u**3 + 30*u**4) / ramp
    edd = sign**2 * (60*u - 180*u**2 + 120*u**3) / ramp**2
    w = 2 * math.pi * frequency
    s, c = math.sin(w*t), math.cos(w*t)
    return (amplitude*envelope*s,
            amplitude*(ed*s + envelope*w*c),
            amplitude*(edd*s + 2*ed*w*c - envelope*w*w*s))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    joints = {int(k): v for k, v in config['joints'].items()}
    for motor_id, settings in joints.items():
        if settings.get('q0') is None:
            raise ValueError(f'Joint {motor_id}: set explicit q0 from the current pose before generating')
        for key in ['q0', 'kp', 'kd']:
            if not isinstance(settings.get(key), (int, float)) or not math.isfinite(settings[key]):
                raise ValueError(f'Joint {motor_id}: explicit finite {key} required')
    ramp, cycles = config['ramp_seconds'], config['cycles']
    if not math.isfinite(ramp) or ramp < 2 or not isinstance(cycles, int) or cycles < 3:
        raise ValueError('ramp_seconds >= 2 and integer cycles >= 3 required')
    selected = config['scan_joints']
    if not selected or len(set(selected)) != len(selected) or any(j not in joints for j in selected):
        raise ValueError('scan_joints must be distinct configured IDs')
    # This is a precomputed task: settle is an explicit fixed hold, not a feedback condition.
    settle = config.get('settle_seconds', 2.0)
    if not math.isfinite(settle) or settle < 0:
        raise ValueError('settle_seconds must be finite and nonnegative')
    settle_frames = math.ceil(settle/0.02)
    points = [(j,a,f,cycles/f) for j in selected for a in AMPLITUDES for f in FREQUENCIES]
    frame_counts = [math.ceil((2*ramp+h)/0.02) for _,_,_,h in points]
    count = settle_frames*(len(points)+1)+sum(frame_counts)
    header = dict(version=1, period_ms=20, duration_ms=count*20, motor_ids=list(joints),
                  start_tolerance_rad=config['settle_position_rad'],
                  max_tracking_error_rad=config['max_tracking_error_rad'],
                  max_temperature_c=config['max_temperature_c'])
    if header['duration_ms'] > 1800000:
        raise ValueError('Task exceeds 30 minutes; generate smaller batches')

    def frames():
        index = 0
        def row(index, active=None, offset=0, velocity=0):
            return dict(at_ms=index*20, motors=[dict(motor_id=j, p=v['q0']+(offset if j == active else 0),
                        v=velocity if j == active else 0, tau=0, kp=v['kp'], kd=v['kd']) for j,v in joints.items()])
        for _ in range(settle_frames):
            yield row(index); index += 1
        for (j,a,f,hold), n in zip(points, frame_counts):
            for i in range(n):
                p,v,_ = trajectory(i*.02,a,f,ramp,hold)
                yield row(index,j,p,v); index += 1
            for _ in range(settle_frames):
                yield row(index); index += 1
    result = write_task(args.output, header, frames())
    print(json.dumps(dict(result, file=str(args.output), points=len(points), duration_ms=count*20), indent=2))
    print('Generated only. Upload validates the file; a separate start command is required.')


if __name__ == '__main__':
    main()
