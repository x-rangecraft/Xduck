from pathlib import Path
import sys,json
import numpy as np,torch
from omegaconf import OmegaConf
R=Path(__file__).resolve().parent;P=R/'project';sys.path.insert(0,str(P/'src'));sys.path.insert(0,str(P/'scripts'))
import train_rsl_rl as E
from unilab.envs.locomotion.xduck_gf43x40.pure_reward import reward_rates,potential,stand_pose_mask
from unilab.algos.torch.pure_distribution import PureGaussian
checks={};torch.set_num_threads(2);torch.manual_seed(42112)
base=dict(up=np.ones(1),height_ratio=np.ones(1),linvel=np.zeros((1,3)),gyro=np.zeros((1,3)),dof_vel=np.zeros((1,14)),action_delta=np.zeros((1,14)),phase='settle',joint_error=np.zeros((1,14)),head_up=np.ones(1),goal_ratio=1.,nominal_height=.319637239)
def score(**changes):return reward_rates(**(base|changes))
def total(t):return float(sum(t.values())[0])
clean=score();low=score(height_ratio=np.array([.8]));err=np.zeros((1,14));err[:,5]=np.deg2rad(50);head=score(joint_error=err,head_up=np.array([np.cos(np.deg2rad(50))]));err2=np.zeros((1,14));err2[:,[0,9]]=np.deg2rad(26);err2[:,[1,10]]=np.deg2rad(12.5);splay=score(joint_error=err2)
checks['low_height_no_longer_gets_full_height_credit']=total(clean)>total(low)+1
checks['low_head_receives_less_goal_and_quiet_credit']=total(clean)>total(head)+1 and head['quiet_bonus'][0]<.05
checks['splayed_hips_receive_dense_cost']=splay['hip_alignment'][0]<0 and total(clean)>total(splay)
checks['wrong_pose_cannot_collect_full_quiet_reward']=low['quiet_bonus'][0]<.05 and splay['quiet_bonus'][0]<clean['quiet_bonus'][0]
checks['final_pose_gate_rejects_low_head_height_and_splayed_hips']=bool(stand_pose_mask(np.array([.319637239]),.319637239,np.ones(1),np.ones(1),np.zeros((1,14)))[0]) and not stand_pose_mask(np.array([.255]),.319637239,np.ones(1),np.ones(1),np.zeros((1,14)))[0] and not stand_pose_mask(np.array([.319637239]),.319637239,np.ones(1),np.ones(1),err)[0] and not stand_pose_mask(np.array([.319637239]),.319637239,np.ones(1),np.ones(1),err2)[0]
moving=score(phase='discovery',linvel=np.array([[.4,0,0]]),gyro=np.array([[0,2.,0]]),dof_vel=np.ones((1,14))*4,action_delta=np.ones((1,14)))
checks['discovery_has_no_velocity_or_smoothness_cost']=all(np.all(moving[k]==0) for k in ['linear_velocity_cost','action_rate','joint_speed','angular_speed'])
distribution_results=[]
for sigma in [.1,1.,3.,15.]:
 dist=PureGaussian(1,init_std=sigma,std_type='log');dist.update(torch.zeros((100000,1)))
 ent=dist.entropy.mean();ent.backward();distribution_results.append(dict(sigma=sigma,raw_entropy=float(dist._distribution.entropy().mean()),bounded_increment_entropy=float(ent.detach()),log_std_gradient=float(dist.log_std_param.grad[0])))
checks['large_latent_std_no_longer_wins_entropy_reward']=distribution_results[-1]['bounded_increment_entropy']<distribution_results[1]['bounded_increment_entropy'] and distribution_results[-1]['log_std_gradient']<0
d=PureGaussian(14,init_std=1.,std_type='log');mu=torch.randn((32,14));d.update(mu);raw=d.sample();expected=torch.distributions.Normal(mu,torch.ones_like(mu)).log_prob(raw).sum(-1);checks['raw_action_log_prob_contract_unchanged']=bool(torch.allclose(d.log_prob(raw),expected,atol=1e-6))
E.ensure_registries();c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');e=E.create_env(c,num_envs=32,env_cfg_override=E.build_ppo_env_cfg_override(c))
try:
 e.init_state();checks['actor_61_and_no_teacher']=e.state.obs['obs'].shape==(32,61) and not hasattr(e,'walk_actor') and not hasattr(e,'prepare_actor')
 for _ in range(32):
  s=e.step(np.random.default_rng(42112).normal(0,1,(32,14)).astype(np.float32));assert np.isfinite(s.reward).all()
 checks['finite_physical_reward_and_head_pose_sensor']=True
 from unilab.training.rsl_rl import RslRlVecEnvWrapper
 wrapper=RslRlVecEnvWrapper(e,device='cuda:0');tc=E.normalize_ppo_train_cfg(OmegaConf.to_container(c.algo,resolve=True));E.apply_ppo_runtime_flags(tc,c,training_enabled=True)
 runner=E.OnPolicyRunner(wrapper,tc,log_dir=None,device='cuda:0');actor=runner.alg.actor;assert isinstance(actor.distribution,PureGaussian)
 before={k:p.detach().clone() for k,p in actor.named_parameters()}
 x=torch.as_tensor(e.state.obs['obs'],device='cuda:0',dtype=torch.float32);mean=actor.mlp(actor.obs_normalizer(x));actor.distribution.update(mean);loss=.001*mean.square().mean()-.01*actor.distribution.entropy.mean();loss.backward()
 checks['native_ppo_uses_repaired_entropy_with_finite_gradients']=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
 checks['native_ppo_validation_did_not_update_model']=all(torch.equal(before[k],p.detach()) for k,p in actor.named_parameters())
finally:e.close()
assert all(checks.values()),checks
out=dict(passed=True,checks=checks,isolated_fixture_scores=dict(correct_stand=total(clean),low_height=total(low),low_head=total(head),splayed_hips=total(splay)),distribution_results=distribution_results,ppo_updates=0,training_remains_paused=True,limitations='Synthetic posture score and entropy tests plus finite physical environment steps; no repaired policy has been trained or qualified.')
encoded=json.dumps(out,indent=2,default=lambda x:x.item());(R/'REPAIR_VERIFICATION.json').write_text(encoded+'\n');print(encoded,flush=True)
