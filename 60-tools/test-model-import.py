"""Real tensor checkpoint → ONNX tests; never contacts hardware.
Run with a Python environment containing torch, onnx, onnxruntime, numpy.
"""
import sys
sys.dont_write_bytecode = True
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import onnx
import onnxruntime as ort

ROOT = Path(os.environ.get("XDUCK_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("model_import", ROOT / "10-source/rk3566/microduck/robotd/src/model_import.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

class ModelImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "policy.pt"
        self.target = Path(self.directory.name) / "policy.onnx"
        torch.manual_seed(3)
        self.actor = torch.nn.Sequential(torch.nn.Linear(61, 16), torch.nn.ELU(), torch.nn.Linear(16, 14)).eval()

    def run_checkpoint(self, checkpoint):
        torch.save(checkpoint, self.source)
        converter.convert(str(self.source), str(self.target), "elu")
        obs = np.random.default_rng(12).normal(size=(1, 61)).astype(np.float32)
        output = ort.InferenceSession(str(self.target), providers=["CPUExecutionProvider"]).run(None, {"obs":obs})[0]
        return obs, output

    def test_isaac_legacy_actor_critic_checkpoint(self):
        state = {"actor." + k:v for k,v in self.actor.state_dict().items()}
        state["std"] = torch.ones(14)
        obs, output = self.run_checkpoint({"model_state_dict":state})
        with torch.inference_mode():
            np.testing.assert_allclose(output, self.actor(torch.from_numpy(obs)).numpy(), rtol=1e-4, atol=1e-5)

    def test_unilab_rsl5_with_real_normalizer(self):
        # Use the exact RSL-RL state-dict layout without making the converter test depend on
        # rsl_rl being installed in robotd's production model environment.
        mean = torch.randn(1, 61)
        std = torch.rand(1, 61) + 0.1
        normalizer = {
            "_mean": mean,
            "_var": std.square(),
            "_std": std,
            "count": torch.tensor(30.0),
        }
        state = {"mlp."+k:v for k,v in self.actor.state_dict().items()}
        state.update({"obs_normalizer."+k:v for k,v in normalizer.items()})
        state["distribution.std_param"] = torch.ones(14)
        obs, output = self.run_checkpoint({"actor_state_dict":state})
        with torch.inference_mode():
            normalized = (torch.from_numpy(obs) - mean) / (std + 0.01)
            np.testing.assert_allclose(output, self.actor(normalized).numpy(), rtol=1e-4, atol=1e-5)
        graph = onnx.load(self.target).graph
        self.assertIn("Sub", {node.op_type for node in graph.node})
        self.assertIn("Div", {node.op_type for node in graph.node})
        self.assertIn("mean", {value.name for value in graph.initializer})

    def test_invalid_contract_and_unsupported_networks_are_rejected(self):
        state = {"actor."+k:v for k,v in self.actor.state_dict().items()}
        cases = []
        wrong_input = dict(state); wrong_input["actor.0.weight"] = torch.zeros(16, 60)
        cases.append((wrong_input, "61"))
        wrong_output = dict(state); wrong_output["actor.2.weight"] = torch.zeros(13, 16); wrong_output["actor.2.bias"] = torch.zeros(13)
        cases.append((wrong_output, "14"))
        bad_values = {k:v.clone() for k,v in state.items()}; bad_values["actor.0.weight"][0,0] = float("nan")
        cases.append((bad_values, "NaN/Inf"))
        recurrent = dict(state); recurrent["memory_a.rnn.weight_ih_l0"] = torch.ones(2,2)
        cases.append((recurrent, "不支持"))
        for weights, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError,message):
                    self.run_checkpoint({"model_state_dict":weights})
                self.assertFalse(self.target.exists())

    def test_unknown_checkpoint_and_corrupt_file_fail(self):
        with self.assertRaisesRegex(ValueError,"未找到"):
            self.run_checkpoint({"mystery":torch.ones(2)})
        self.source.write_bytes(b"not a torch checkpoint")
        with self.assertRaises(Exception):
            converter.convert(str(self.source), str(self.target), "elu")
        self.assertFalse(self.target.exists())

    def test_built_in_locomotion_policy_uses_raw_action_and_requested_mit_gains(self):
        runtime_policy = ROOT / "10-source/rk3566/microduck/robotd/src/default_locomotion_policy.py"
        sdk_policy = ROOT / "00-docs/XDUCK_POLICY_SDK/policy.py"
        if sdk_policy.exists():
            self.assertEqual(runtime_policy.read_bytes(), sdk_policy.read_bytes())
        policy_spec = importlib.util.spec_from_file_location("default_policy", runtime_policy)
        policy_module = importlib.util.module_from_spec(policy_spec)
        policy_spec.loader.exec_module(policy_module)
        frame = {
            "positions": policy_module.HOME.tolist(),
            "velocities": [0.0] * 15,
            "imu": {"gyro": [0.0] * 3, "gravity": [0.0, 0.0, -1.0]},
            "command": {"twist": [0.1, 0.0, 0.0], "head": [0.0] * 4, "body": [0.0] * 3},
            "policy_context": {"action": "walk", "phase": None, "body_active": False},
        }
        policy = policy_module.Policy()
        policy.reset({"joint_names": policy_module.JOINTS}, frame)
        action = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
        targets = policy.postprocess({"actions": action.reshape(1, 14)}, frame)
        self.assertTrue(all(target["kp"] == 60.0 and target["kd"] == 4.0 for target in targets.values()))
        next_obs = policy.preprocess(frame, None)["obs"].reshape(61)
        np.testing.assert_array_equal(next_obs[34:48], action)

if __name__ == "__main__":
    unittest.main()
