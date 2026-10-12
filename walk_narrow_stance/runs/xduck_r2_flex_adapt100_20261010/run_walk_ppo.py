from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np,torch
from omegaconf import OmegaConf

R = Path(__file__).resolve().parent
sys.path.insert(0,str(R/'project'/'src'))
sys.path.insert(0,str(R/'project'/'scripts'))
import train_rsl_rl as E
from unilab.training.rsl_rl import normalize_ppo_train_cfg,RslRlVecEnvWrapper,attach_checkpoint_progress
from rsl_rl.runners import OnPolicyRunner

ap = argparse.ArgumentParser()
ap.add_argument('--smoke',action='store_true')
ap.add_argument('--iterations',type=int,default=100)
ap.add_argument('--parent',default='ready_start_r1_1000.pt')
ap.add_argument('--config',default='PPO_CONFIG.yaml')
ap.add_argument('--logdir',default='ppo_run')
ap.add_argument('--max-std',type=float,default=None)
ap.add_argument('--exploration-std',type=float,default=None)
args = ap.parse_args()
torch.set_num_threads(2)
cfg = OmegaConf.load(R/args.config)
torch.manual_seed(int(cfg.algo.seed))
np.random.seed(int(cfg.algo.seed))
if args.smoke:
    cfg.algo.num_envs = 16
E.ensure_registries()
env = E.create_env(cfg,num_envs=cfg.algo.num_envs,env_cfg_override=E.build_ppo_env_cfg_override(cfg))
before = hashlib.sha256((R/'xc_standing.onnx').read_bytes()).hexdigest()
try:
    wrapped = RslRlVecEnvWrapper(env,device='cuda')
    train_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo,resolve=True))
    E.apply_ppo_runtime_flags(train_cfg,cfg,training_enabled=True)
    train_cfg.setdefault('runner',{})['logger'] = 'none' if args.smoke else 'tensorboard'
    train_cfg['logger'] = train_cfg['runner']['logger']
    runner = OnPolicyRunner(wrapped,train_cfg,log_dir=None if args.smoke else str(R/args.logdir),device='cuda')
    parent = torch.load(R/args.parent,map_location='cuda',weights_only=False)
    # Load Walk weights only. The newly constructed PPO optimizer remains fresh.
    runner.alg.actor.load_state_dict(parent['actor_state_dict'],strict=True)
    runner.alg.critic.load_state_dict(parent['critic_state_dict'],strict=True)
    runner.alg.actor.obs_normalizer.eval()
    if args.max_std is not None:
        with torch.no_grad():
            runner.alg.actor.distribution.std_param.clamp_(max=args.max_std)
            if args.exploration_std is not None:runner.alg.actor.distribution.std_param.fill_(args.exploration_std)
        runner.alg.actor.distribution.std_param.requires_grad_(False)
        print('FROZEN_EXPLORATION_STD',runner.alg.actor.distribution.std_param.detach().cpu().tolist(),flush=True)
    assert len(runner.alg.optimizer.state)==0
    if bool(cfg.env.flexible_dr_enabled):
        validation_obs,_=wrapped.reset()
        assert validation_obs['actor'].shape==(cfg.algo.num_envs,61)
        joints=env._flex_joint_ids;dofs=env._flex_dof_ids
        k=env._backend._device_model.jnt_stiffness.numpy()[:,joints]/env._nominal_flex_stiffness[joints]
        d=env._backend._device_model.dof_damping.numpy()[:,dofs]/env._nominal_flex_damping[dofs]
        assert len(joints)==48 and .8999<k.min()<k.max()<1.1001 and .8999<d.min()<d.max()<1.1001
        phi=env._backend.get_physics_state();passive=phi[:,1+env._passive_qpos_indices]
        assert abs(passive).max()>1e-6
        (R/('effective_dr_smoke.json' if args.smoke else 'effective_dr_training.json')).write_text(json.dumps(dict(passive_flex_coordinates=48,actual_stiffness_ratio=[float(k.min()),float(k.max())],actual_damping_ratio=[float(d.min()),float(d.max())],sole_angle_range_deg=np.degrees([env._sole_angles.min(),env._sole_angles.max()]).tolist(),restored_nonzero_passive_equilibrium=float(abs(passive).max()),actor_dim=61,standing599_calls=0),indent=2))
    attach_checkpoint_progress(env,runner)
    if args.smoke:
        obs,_ = wrapped.reset()
        assert obs['actor'].shape==(16,61)
        np.testing.assert_allclose(obs['actor'][:,48:51].cpu().numpy(),env._dir_commands[:,:3],atol=1e-6)
        goal = np.asarray(json.loads((R/'NEW_STANDING_TARGET.json').read_text())['target_rad'])
        np.testing.assert_allclose(env._current_action_target,np.broadcast_to(goal,(16,14)),atol=3e-7)
        np.testing.assert_allclose(env._previous_action_target,env._current_action_target,atol=3e-7)
        expected_history = (goal-env.default_angles)/env._physical_scales
        np.testing.assert_allclose(obs['actor'][:,34:48].cpu().numpy(),
                                   np.broadcast_to(expected_history,(16,14)),atol=3e-6)
        for i in range(360):
            with torch.inference_mode():
                action = runner.alg.actor.mlp(runner.alg.actor.obs_normalizer(obs['actor']))
            obs,reward,done,info = wrapped.step(action)
            assert action.shape==(16,14)
            assert torch.isfinite(reward).all() and torch.isfinite(obs['actor']).all()
            np.testing.assert_allclose(obs['actor'][:,48:51].cpu().numpy(),env._dir_commands[:,:3],atol=1e-6)
            if (i+1)%120==0:
                print('SMOKE',i+1,float(reward.mean()),flush=True)
        (R/'smoke_validation.json').write_text(json.dumps(dict(steps=360,actor_dim=61,action_dim=14,
             rewards_finite=True,reset_exercised=True,optimizer_fresh=True,
             standing_policy_called=False),indent=2))
    else:
        start = time.time()
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        (R/(args.logdir+'_completion.json')).write_text(json.dumps(dict(
            iterations=args.iterations,elapsed_s=time.time()-start,standing_policy_called=False,
            stand599_unchanged=before==hashlib.sha256((R/'xc_standing.onnx').read_bytes()).hexdigest(),
            requires_closed_loop_validation=True),indent=2))
        print('WALK_PPO_DONE',flush=True)
    assert before==hashlib.sha256((R/'xc_standing.onnx').read_bytes()).hexdigest()
finally:
    env.close()
