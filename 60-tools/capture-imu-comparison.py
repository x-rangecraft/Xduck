#!/usr/bin/env python3
"""Read-only, concurrent 50 Hz capture of Jetson N100 and Xduck BMI088 telemetry."""

import argparse
import csv
import json
import shlex
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
N100_READER = r'''
import rclpy
import time
from common.msg import ImuData
from rclpy.qos import qos_profile_sensor_data

def on_sample(msg):
    global last_bin
    stamp_ns = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
    sample_bin = stamp_ns // 20000000
    if sample_bin == last_bin:
        return
    last_bin = sample_bin
    values = (
        time.time_ns(), stamp_ns,
        msg.sequence, int(msg.orientation_valid), *msg.orientation_xyzw,
        *msg.angular_velocity_rps, *msg.linear_acceleration_mps2,
    )
    print(','.join(map(str, values)), flush=True)

last_bin = None
rclpy.init()
node = rclpy.create_node('n100_comparison_capture')
node.create_subscription(ImuData, '/body_imu/topic/data', on_sample, qos_profile_sensor_data)
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
'''


def read_n100(proc, path, counts):
    fields = ['host_wall_ns', 'ros_stamp_ns', 'sequence', 'orientation_valid',
              'qx', 'qy', 'qz', 'qw', 'gx_rad_s', 'gy_rad_s', 'gz_rad_s',
              'ax_m_s2', 'ay_m_s2', 'az_m_s2']
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for line in proc.stdout:
            parts = line.strip().split(',')
            if len(parts) == len(fields):
                writer.writerow(parts)
                counts['n100'] += 1


def read_rk(proc, path, counts):
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['receive_wall_ns', 'robot_t_s', 'loop_ticks',
                         'roll_rad', 'pitch_rad', 'yaw_rad'])
        for line in proc.stdout:
            received = time.time_ns()
            try:
                event = json.loads(line)
                if event.get('method') != 'robot.state':
                    continue
                state = event['params']
                writer.writerow([received, state['t'], state['loop']['ticks'],
                                 *state['imu_rpy']])
                counts['rk3566'] += 1
            except (ValueError, KeyError, TypeError):
                continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=20)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 5:
        parser.error('--seconds must be at least 5')
    args.out.mkdir(parents=True, exist_ok=True)

    n100_shell = ('source /opt/ros/humble/setup.bash && '
                  'source /home/jetson/EragonROS/install/setup.bash && '
                  'python3 -u -c ' + shlex.quote(N100_READER))
    ssh_options = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes']
    commands = {
        'n100': ['ssh', *ssh_options, 'jetson@192.168.50.166',
                 'bash -lc ' + shlex.quote(n100_shell)],
        'rk3566': [str(ROOT / '60-tools/ssh-rk3566.sh'),
                   'robotctl monitor --hz 0 --json'],
    }
    procs = {name: subprocess.Popen(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
             for name, command in commands.items()}
    counts = {'n100': 0, 'rk3566': 0}
    threads = [
        threading.Thread(target=read_n100, args=(procs['n100'], args.out / 'n100-50hz.csv', counts)),
        threading.Thread(target=read_rk, args=(procs['rk3566'], args.out / 'rk3566-bmi088-50hz.csv', counts)),
    ]
    start_ns = time.time_ns()
    for thread in threads:
        thread.start()
    try:
        time.sleep(args.seconds)
    finally:
        for proc in procs.values():
            proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for thread in threads:
            thread.join(timeout=3)
    summary = {'start_wall_ns': start_ns, 'end_wall_ns': time.time_ns(),
               'requested_seconds': args.seconds, 'samples': counts,
               'sources': {'n100': '/body_imu/topic/data; 20 ms timestamp gate from 100 Hz',
                           'rk3566': 'robotctl monitor --hz 0 --json; every 50 Hz control tick, BMI088 fused RPY'}}
    (args.out / 'capture.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if min(counts.values()) == 0:
        raise SystemExit('one sensor produced no samples')


if __name__ == '__main__':
    main()
