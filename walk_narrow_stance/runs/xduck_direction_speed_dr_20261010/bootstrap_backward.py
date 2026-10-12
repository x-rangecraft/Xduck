from common import *
import torch,copy,json
from pathlib import Path
torch.set_num_threads(2);torch.manual_seed(1510091)
parent=torch.load(R/'ppo_direction_run/model_150.pt',map_location='cpu',weights_only=False);sd=parent['actor_state_dict']
class Actor(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',sd['obs_normalizer._mean'].clone());self.register_buffer('std',sd['obs_normalizer._std'].clone())
  self.net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14))
  self.net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')})
 def forward(self,x):return self.net((x-self.mean)/(self.std+.01))
z=np.load(R/'direction_anchor.npz');parts=[(z['obs'],z['mean'],z['mask'],z['phase'])]
z=np.load(R/'backward_mix_trace.npz');selected=[]
for r in json.loads((R/'backward_mix_results.json').read_text())['records']:
 if not(r['velocity_mean']>.045 and r['min1s_velocity']>.025 and r['tilt_max_deg']<20 and r['min_height']>.25 and abs(r['yaw_change_deg'])<90 and 202<r['steady_width_mm']<222 and all(20<p<48 for p in r['side_peak_median_mm'])):continue
 j=r['trace_index'];o=z['obs'][:,j];y=z['raw'][:,j]
 parts.append((o,y,np.ones_like(y),np.full(len(o),3)));selected.append(r)
assert len(selected)>=2,'No sufficiently valid physical backward examples'
# Preserve the current PPO's physically successful turn and forward decoder.
z=np.load(R/'direction_mid_125_trace.npz')
for r in json.loads((R/'direction_mid_125_results.json').read_text())['records']:
 if not r['rough_tracking_pass']:continue
 j=r['trace_index'];o=z['obs'][:,j];y=z['raw'][:,j]
 parts.append((o,y,np.ones_like(y),np.ones(len(o),int)))
X=np.concatenate([p[0] for p in parts]).astype(np.float32);Y=np.concatenate([p[1] for p in parts]).astype(np.float32);M=np.concatenate([p[2] for p in parts]).astype(np.float32);P=np.concatenate([p[3] for p in parts]).astype(int)
np.savez_compressed(R/'direction_anchor_r2.npz',obs=X,mean=Y,mask=M,phase=P)
x=torch.tensor(X);y=torch.tensor(Y);m=torch.tensor(M);p=torch.tensor(P);model=Actor();opt=torch.optim.Adam(model.net.parameters(),lr=8e-5)
pools=[torch.nonzero(p==i).flatten() for i in range(4)];weights=[2.,1.,.5,2.]
for step in range(800):
 losses=[]
 for i,pool in enumerate(pools):
  ids=pool[torch.randint(len(pool),(128,))];mask=m[ids];loss=(((model(x[ids])-y[ids])/.35)**2*mask).sum()/mask.sum().clamp_min(1);losses.append(loss)
 total=sum(w*v for w,v in zip(weights,losses));opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.net.parameters(),1.);opt.step()
 if (step+1)%200==0:print('BOOTSTRAP',step+1,[round(float(v.detach()),5) for v in losses],flush=True)
ck=copy.deepcopy(parent)
for k,v in model.net.state_dict().items():ck['actor_state_dict']['mlp.'+k]=v.clone()
ck.pop('optimizer_state_dict',None)
for k,v in sd.items():
 if not k.startswith('mlp.'):assert torch.equal(v,ck['actor_state_dict'][k]),k
ck.setdefault('infos',{})['backward_bootstrap']=dict(teacher='physical clean-leg/current-width mixture',selected_indices=[r['trace_index'] for r in selected],standing599_calls=0,not_final_accepted=True)
torch.save(ck,R/'backward_bootstrap.pt')
model.eval();torch.onnx.export(model,torch.zeros(1,61),str(R/'backward_bootstrap.onnx'),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
ix=np.random.default_rng(1510091).choice(len(X),128,replace=False)
with torch.no_grad():expected=model(x[ix]).numpy()
actual=np.concatenate([actor('backward_bootstrap.onnx').run(None,{'obs':o[None]})[0] for o in X[ix]]);err=float(abs(expected-actual).max());assert err<2e-5
cfg=OmegaConf.load(R/'PPO_DIRECTION_CONFIG.yaml');cfg.algo.seed=1510091;cfg.env.direction_seed=1510091;cfg.env.newstand_seed=1510091
cfg.algo.algorithm.anchor_dataset=str(R/'direction_anchor_r2.npz');cfg.algo.algorithm.anchor_weight=.04
OmegaConf.save(cfg,R/'PPO_DIRECTION_R2_CONFIG.yaml')
(R/'backward_bootstrap_manifest.json').write_text(json.dumps(dict(selected=selected,phase_counts={str(i):int((P==i).sum()) for i in range(4)},onnx_error=err,teacher_speed_below_final_target=True,normalizer_critic_std_unchanged=True,standing599_calls=0),indent=2))
print('BACKWARD_BOOTSTRAP_DONE',err,flush=True)
