"""Real tensor checkpoint → ONNX tests; never contacts hardware.
Run with a Python environment containing torch, onnx, onnxruntime, numpy.
"""
import sys
sys.dont_write_bytecode = True
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
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
        # Compare against RSL-RL's actual module, including the normalization epsilon.
        from rsl_rl.modules import EmpiricalNormalization
        normalizer = EmpiricalNormalization(61).eval()
        normalizer.update(torch.randn(30, 61) * 2 + 1)
        state = {"mlp."+k:v for k,v in self.actor.state_dict().items()}
        state.update({"obs_normalizer."+k:v for k,v in normalizer.state_dict().items()})
        state["distribution.std_param"] = torch.ones(14)
        obs, output = self.run_checkpoint({"actor_state_dict":state})
        with torch.inference_mode():
            np.testing.assert_allclose(output, self.actor(normalizer(torch.from_numpy(obs))).numpy(), rtol=1e-4, atol=1e-5)

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

if __name__ == "__main__":
    unittest.main()
