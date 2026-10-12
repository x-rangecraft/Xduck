from common import *
import torch,copy,json
torch.set_num_threads(2);torch.manual_seed(1530091)
parent=torch.load(R/'ppo_direction_r2_run/model_179.pt',map_location='cpu',weights_only=False);sd=parent['actor_state_dict']
class Actor(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',sd['obs_normalizer._mean'].clone());self.register_buffer('std',sd['obs_normalizer._std'].clone())
  self.net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14))
  self.net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')})
 def forward(self,x):return self.net((x-self.mean)/(self.std+.01))
z=np.load(R/'direction_anchor.npz');parts=[(z['obs'],z['mean'],z['mask'],z['phase'])]
z=np.load(R/'r2_cold_131_trace.npz');keep=[]
for r in json.loads((R/'r2_cold_131_results.json').read_text())['records']:
 if not r['rough_tracking_pass']:continue
 if r['command'][0]:
  if abs(r['speed_error_mps'])>.05:continue
  if r['scenario']!='backward' and not(.42<=r['same_foot_period_s']<=.58):continue
 else:
  if r['pure_turn_residual_speed_mps']>.09:continue
 j=r['trace_index'];o=z['obs'][:,j];y=z['raw'][:,j];phase=3 if r['scenario']=='backward' else 1
 parts.append((o,y,np.ones_like(y),np.full(len(o),phase)));keep.append(r)
z=np.load(R/'turn_teacher_trace.npz');selected=[]
for r in json.loads((R/'turn_teacher_results.json').read_text())['records']:
 if not(r['rough_tracking_pass'] and r['pure_turn_residual_speed_mps']<.09 and 202<r['steady_width_mm']<222 and all(8<p<45 for p in r['side_peak_median_mm'])):continue
 j=r['trace_index'];o=z['obs'][:150,j];y=z['raw'][:150,j]
 parts.append((o,y,np.ones_like(y),np.full(len(o),4)));selected.append(r)
assert len(selected)>=6,'Too few valid physical pure-turn startup examples'
X=np.concatenate([p[0] for p in parts]).astype(np.float32);Y=np.concatenate([p[1] for p in parts]).astype(np.float32);M=np.concatenate([p[2] for p in parts]).astype(np.float32);P=np.concatenate([p[3] for p in parts]).astype(int)
np.savez_compressed(R/'direction_anchor_r3.npz',obs=X,mean=Y,mask=M,phase=P)
x=torch.tensor(X);y=torch.tensor(Y);m=torch.tensor(M);p=torch.tensor(P);model=Actor();opt=torch.optim.Adam(model.net.parameters(),lr=6e-5)
pools=[torch.nonzero(p==i).flatten() for i in range(5)];weights=[2.,1.,.25,1.,1.5]
assert all(len(pool) for pool in pools)
for step in range(600):
 losses=[]
 for i,pool in enumerate(pools):
  ids=pool[torch.randint(len(pool),(128,))];mask=m[ids];loss=(((model(x[ids])-y[ids])/.35)**2*mask).sum()/mask.sum().clamp_min(1);losses.append(loss)
 total=sum(w*v for w,v in zip(weights,losses));opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.net.parameters(),1.);opt.step()
 if (step+1)%200==0:print('BOOTSTRAP_TURN',step+1,[round(float(v.detach()),5) for v in losses],flush=True)
ck=copy.deepcopy(parent)
for k,v in model.net.state_dict().items():ck['actor_state_dict']['mlp.'+k]=v.clone()
ck.pop('optimizer_state_dict',None)
for k,v in sd.items():
 if not k.startswith('mlp.'):assert torch.equal(v,ck['actor_state_dict'][k]),k
torch.save(ck,R/'turn_bootstrap.pt');model.eval()
torch.onnx.export(model,torch.zeros(1,61),str(R/'turn_bootstrap.onnx'),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
ix=np.random.default_rng(1530091).choice(len(X),128,replace=False)
with torch.no_grad():expected=model(x[ix]).numpy()
actual=np.concatenate([actor('turn_bootstrap.onnx').run(None,{'obs':o[None]})[0] for o in X[ix]]);err=float(abs(expected-actual).max());assert err<2e-5
cfg=OmegaConf.load(R/'PPO_DIRECTION_R2_CONFIG.yaml');cfg.algo.seed=1530091;cfg.env.direction_seed=1530091;cfg.env.newstand_seed=1530091
cfg.algo.algorithm.anchor_dataset=str(R/'direction_anchor_r3.npz');cfg.algo.algorithm.anchor_weight=.035
OmegaConf.save(cfg,R/'PPO_DIRECTION_R3_CONFIG.yaml')
(R/'turn_bootstrap_manifest.json').write_text(json.dumps(dict(selected=selected,retained=keep,phase_counts={str(i):int((P==i).sum()) for i in range(5)},onnx_error=err,normalizer_critic_std_unchanged=True,teacher_modified_forward_input_only_during_first_0p8s=True,recorded_training_commands_are_actual_pure_turn=True,supervised_output_has_no_runtime_input_helper=True,standing599_calls=0,seed131_now_selection_not_final_holdout=True),indent=2))
print('TURN_BOOTSTRAP_DONE',err,flush=True)
