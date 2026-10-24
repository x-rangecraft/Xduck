#!/usr/bin/env python3
"""Read-only, 30-second post-install evidence. Run on the selected device as root."""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def cpu_ticks():
    values = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3] + values[4]


start_cpu = cpu_ticks()
started = time.monotonic()
samples = []
with socket.socket(socket.AF_UNIX) as client:
    client.settimeout(5)
    client.connect('/run/robotd.sock')
    client.sendall((json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'robot.subscribe',
                               'params': {'hz': 2}}) + '\n').encode())
    stream = client.makefile()
    while time.monotonic() - started < 30:
        line = stream.readline()
        if not line:
            raise RuntimeError('robot state stream ended')
        message = json.loads(line)
        if message.get('method') == 'robot.state':
            samples.append(message['params'])
end_cpu = cpu_ticks()
assert len(samples) >= 40, 'Too few state samples'
assert all(s['control_state'] == 'relaxed' for s in samples), 'Device is not relaxed'
assert all(not any(flag & 4 for flag in s['motor_flags']) for s in samples), 'A motor is enabled'
units = {}
for unit in ('updaterd', 'robotd', 'configd', 'btd', 'padd', 'mediad', 'xpad-usbd'):
    pid = int(command('systemctl', 'show', unit, '-p', 'MainPID', '--value'))
    units[unit] = {'active': command('systemctl', 'is-active', unit),
                   'enabled': command('systemctl', 'is-enabled', unit),
                   'exe': os.readlink(f'/proc/{pid}/exe')}
http = http.client.HTTPConnection('127.0.0.1', 8080, timeout=5)
http.request('GET', '/')
response = http.getresponse()
web_status = response.status
response.read()
http.close()
total = end_cpu[0] - start_cpu[0]
result = {
    'hostname': command('hostname'),
    'machine_id': Path('/etc/machine-id').read_text().strip(),
    'soc_serial': Path('/proc/device-tree/serial-number').read_text().strip('\0\n'),
    'kernel': command('uname', '-r'),
    'current_release': os.path.realpath('/opt/robot/daemon/current'),
    'sample_seconds': time.monotonic() - started,
    'sample_count': len(samples),
    'cpu_busy_percent': round(100 * (1 - (end_cpu[1] - start_cpu[1]) / total), 2),
    'loadavg': Path('/proc/loadavg').read_text().strip(),
    'meminfo': {line.split(':')[0]: line.split(':')[1].strip()
                for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith(('MemTotal:', 'MemAvailable:'))},
    'temperature_c': int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000,
    'loop_hz_min': min(s['loop']['hz'] for s in samples),
    'loop_hz_max': max(s['loop']['hz'] for s in samples),
    'missed_tick_delta': samples[-1]['loop']['missed'] - samples[0]['loop']['missed'],
    'motor_enabled_seen': False,
    'motor_online_count_last': sum(bool(f & 2) for f in samples[-1]['motor_flags']),
    'camera_usb_present': Path('/dev/video-xrange-camera').exists(),
    'web_http_status': web_status,
    'units': units,
    'failed_units': command('systemctl', '--failed', '--no-legend', '--no-pager'),
    'health': json.loads(command('robotctl', 'health', '--json')),
    'last_state': samples[-1],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
