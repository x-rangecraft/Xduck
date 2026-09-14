#!/usr/bin/env python3
"""Offline task generator regression. No device connection or control."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'10-source/rk3566/microduck/mediad/webclient/motor-experiment'
class Tests(unittest.TestCase):
    def test_offline_generation_contains_every_frame_and_motor(self):
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'config.json';out=Path(d)/'task.jsonl'
            config.write_text(json.dumps(dict(joints={'2':dict(q0=0,kp=60,kd=4),'13':dict(q0=.1,kp=60,kd=4)},scan_joints=[2],ramp_seconds=2,cycles=3,settle_position_rad=.05,max_tracking_error_rad=.3,max_temperature_c=60)))
            subprocess.run([sys.executable,str(WEB/'joint_sine_sweep.py'),'--config',str(config),'--output',str(out)],check=True,capture_output=True)
            lines=[json.loads(x) for x in out.read_text().splitlines()];header,*rows=lines
            self.assertEqual(len(rows)*20,header['duration_ms'])
            self.assertEqual([r['at_ms'] for r in rows],list(range(0,header['duration_ms'],20)))
            self.assertTrue(all({m['motor_id'] for m in r['motors']}=={2,13} for r in rows))
            self.assertTrue(all(next(m for m in r['motors'] if m['motor_id']==13)['p']==.1 for r in rows))
    def test_no_implicit_q0(self):
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'config.json';config.write_text(json.dumps({'joints':{'2':{'q0':None}}}))
            result=subprocess.run([sys.executable,str(WEB/'joint_sine_sweep.py'),'--config',str(config),'--output',str(Path(d)/'task.jsonl')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('explicit q0',result.stderr)
if __name__=='__main__':unittest.main()
