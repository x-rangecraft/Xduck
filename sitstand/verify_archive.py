"""Check archived policy against the actual frozen physical task; no training."""
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from unilab.training import BackendAdapter, create_env, ensure_registries
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'policy'))
import reference_adapter as adapter
cfg = OmegaConf.create(json.loads((root/'checkpoints/run_config.json').read_text())['config'])
cfg.env.warp_compiled_model_file = str(root/'physics/pure_effective_newstand_linux.mjb')
cfg.env.evaluation_case = 2
ensure_registries()
actor_cfg = OmegaConf.to_container(cfg.algo.actor, resolve=True)
actor_cfg.pop('class_name')
actor = MLPModel(TensorDict({'actor': torch.zeros(16,61)}, batch_size=[16]), {'actor':['actor']}, 'actor', 14, **actor_cfg)
saved = torch.load(root/'checkpoints/model_2499.pt',map_location='cpu',weights_only=False)
actor.load_state_dict(saved['actor_state_dict']); actor.eval()
session = ort.InferenceSession(str(root/'policy/xduck_sitstand_2500_newstand.onnx'), providers=['CPUExecutionProvider'])
env = create_env(cfg,num_envs=16,env_cfg_override=BackendAdapter(cfg,root_dir=root/'unilab').build_task_env_cfg_override())
try:
    env.step_counter = saved['infos']['unilab_progress']['env_steps_per_env']
    env._update_curriculum(); env.init_state()
    np.testing.assert_allclose(env.default_angles,adapter.DEFAULT,atol=1e-7)
    np.testing.assert_allclose(env._target_low,adapter.LOW,atol=1e-7)
    np.testing.assert_allclose(env._target_high,adapter.HIGH,atol=1e-7)
    raw_error = 0.
    for _ in range(16):
        obs = env.state.obs['obs'].copy()
        with torch.inference_mode(): actions = actor(TensorDict({'actor':torch.from_numpy(obs)},batch_size=[16])).numpy()
        onnx_action = session.run(None,{'obs':obs[:1]})[0]
        raw_error=max(raw_error,float(np.max(np.abs(actions[:1]-onnx_action))))
        env.step(actions)
        assert np.isfinite(env.state.obs['obs']).all()
    assert raw_error<1e-5,raw_error
    result={'physical_steps':16,'environments':16,'completed_updates':2500,'env_steps_per_env':env.step_counter,'raw_onnx_max_error':raw_error,'default_and_bounds_match':True,'finite_physics_observations':True,'training_performed':False}
    (root/'outputs/archive_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
finally: env.close()
