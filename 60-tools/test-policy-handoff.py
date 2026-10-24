#!/usr/bin/env python3
"""Run the production worker through suspended-history handoffs, without motor IO.
Optional arguments: actual v6 policy.py and standing.onnx snapshots.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / '10-source/rk3566/microduck/robotd/src/custom_policy_worker.py'
JOINTS = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
          'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
          'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll', 'mouth']
CONTROLLED = JOINTS[:-1]
FRAME = dict(sequence=1, gateway_tick_ms=20, dt=.02, joint_names=JOINTS,
             positions=[0.] * 15, velocities=[.1] * 15, motor_torques_nm=[0.] * 15,
             motor_flags=[7] * 15, motor_temperatures_c=[25.] * 15,
             motor_feedback_age_ms=[0] * 15, attitude_rpy=[0.] * 3,
             imu=dict(gyro=[0., 2.5, 0.], gravity=[.3, 0., -.95], quat=[1., 0., 0., 0.]),
             command=dict(twist=[0.] * 3, head=[0.] * 4, body=[0.] * 3),
             policy_context=dict(action='stand', phase=None, body_active=False))


def targets(value):
    return {name: dict(position=value + i * .001, velocity=0., torque_ff=0., kp=60., kd=4.)
            for i, name in enumerate(CONTROLLED)}


def worker(policy, model):
    proc = subprocess.Popen([sys.executable, '-I', '-u', '-c', WORKER.read_text(), str(policy), str(model)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ready = json.loads(proc.stdout.readline())
    assert ready['ok'], ready
    return proc


def request(proc, action, **fields):
    proc.stdin.write(json.dumps(dict(action=action, frame=FRAME, **fields)) + '\n')
    proc.stdin.flush()
    reply = json.loads(proc.stdout.readline())
    assert reply['ok'], reply
    return reply


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    model = tmp / 'identity.onnx'
    graph = helper.make_graph([helper.make_node('Identity', ['obs'], ['actions'])], 'handoff',
                              [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 14])],
                              [helper.make_tensor_value_info('actions', TensorProto.FLOAT, [1, 14])])
    net = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    net.ir_version = 10
    model.write_bytes(net.SerializeToString())
    policy = tmp / 'consumer.py'
    policy.write_text('''import numpy as np
CONTROLLED = ''' + repr(CONTROLLED) + '''
class Policy:
    def describe(self):
        return dict(api_version=2,period_us=40000,required_sources=["joints"],controlled_joints=CONTROLLED,
                    inputs={"obs":{"dtype":"float32","shape":[1,14]}},
                    outputs={"actions":{"dtype":"float32","shape":[1,14]}})
    def reset(self, robot_info, first_frame):
        self.integrator = np.asarray(first_frame["positions"][:14])
    def preprocess(self, frame, feedback):
        if feedback and feedback.get("handoff"):
            assert frame["positions"] == [0.] * 15, "measured frame was overwritten"
            history = np.asarray([feedback["requested_targets"][n]["position"] for n in CONTROLLED])
            assert np.array_equal(history, self.integrator), "obs and integrator differ"
        return {"obs":self.integrator.astype(np.float32).reshape(1,14)}
    def postprocess(self, outputs, frame):
        self.integrator += .01
        return {n:dict(position=float(self.integrator[i]),velocity=0.,torque_ff=0.,kp=60.,kd=4.) for i,n in enumerate(CONTROLLED)}
''')
    proc = worker(policy, model)
    try:
        request(proc, 'reset')
        for _ in range(5):
            request(proc, 'step')
        for value in (.3, -.2, .8):
            handoff = targets(value)
            request(proc, 'reset', handoff_targets=handoff)
            result = request(proc, 'step')['targets']
            for name in CONTROLLED:
                assert abs(result[name]['position'] - handoff[name]['position'] - .01) < 1e-12
            request(proc, 'step')
        proc.stdin.write(json.dumps(dict(action='reset', frame=FRAME, handoff_targets={})) + '\n')
        proc.stdin.flush()
        assert not json.loads(proc.stdout.readline())['ok'], 'partial handoff was accepted'
    finally:
        proc.kill()
        proc.wait()
print('PASS: production worker replaces suspended state; target history and integrator agree; measured sensors preserved; incomplete handoff rejected')

if len(sys.argv) == 3:
    import onnxruntime as ort
    policy, model = map(Path, sys.argv[1:])
    spec = importlib.util.spec_from_file_location('actual_v6', policy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
    proc = worker(policy, model)
    max_error = 0.
    try:
        request(proc, 'reset')
        for _ in range(5):
            request(proc, 'step')
        for value in (.2, -.1, .6):
            handoff = targets(value)
            for i, name in enumerate(module.TRAINING_JOINTS):
                handoff[name]["position"] = float(np.clip(handoff[name]["position"], module.STAND_TARGET_LOW[i] + .05, module.STAND_TARGET_HIGH[i] - .05))
            # Force the neck far from the frozen worker's preceding targets.
            handoff['neck_pitch']['position'] = 1.1
            request(proc, 'reset', handoff_targets=handoff)
            result = request(proc, 'step')['targets']
            consumer = module.Policy()
            first = dict(FRAME, positions=[handoff[n]['position'] if n in handoff else 0. for n in JOINTS])
            consumer.reset(dict(joint_names=JOINTS), first)
            inputs = consumer.preprocess(FRAME, dict(outputs={}, requested_targets=handoff, handoff=True))
            history = (np.asarray([handoff[n]['position'] for n in module.TRAINING_JOINTS]) - module.STAND_REFERENCE) / module.STAND_HISTORY_SCALE
            np.testing.assert_allclose(inputs['obs'][0, 34:48], history, atol=1e-6)
            output = session.run(None, inputs)[0]
            expected = consumer.postprocess(dict(actions=output), FRAME)
            for i, name in enumerate(module.TRAINING_JOINTS):
                max_error = max(max_error, abs(expected[name]['position'] - result[name]['position']))
                assert abs(result[name]['position'] - handoff[name]['position']) <= module.STAND_INCREMENT[i] + 1e-8
        assert max_error < 1e-8, max_error
    finally:
        proc.kill()
        proc.wait()
    print(f'PASS: actual v6 + ONNX, three handoffs, reference max error {max_error:.3g} rad; first increments bounded')
