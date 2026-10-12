from common import *
import torch,json,copy,hashlib
from pathlib import Path
torch.set_num_threads(2)
parent=torch.load(R/'ppo_clean_r3_run/model_80.pt',map_location='cpu',weights_only=False);sd=parent['actor_state_dict']
class Actor(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',sd['obs_normalizer._mean'].clone());self.register_buffer('std',sd['obs_normalizer._std'].clone())
  self.net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14))
  self.net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')})
 def forward(self,x):return self.net((x-self.mean)/(self.std+.01))

model=Actor().eval()
parts=[];selected=[]
for prefix in ['startup_connection_probe','startup_brace_pd_probe']:
 z=np.load(R/(prefix+'_trace.npz'));records=json.loads((R/(prefix+'_results.json')).read_text())['records']
 for rec in records:
  if rec['model'] not in ['brace200','pd200']:continue
  if not(rec['healthy'] and rec['natural_width'] and all(26<h<38 for h in rec['side_peak_median_mm']) and .16<rec['velocity_mean']<.245 and rec['startup_2p5_vertical_speed_max']<.40 and rec['shallow_startup_landings']<=2):continue
  j=rec['trace_index'];parts.append((z['obs'][:100,j],z['raw'][:100,j],20.))
  parts.append((z['obs'][100:,j],z['raw'][100:,j],4.));selected.append(dict(prefix=prefix,trace_index=j,record=rec))
assert selected,'No qualified teacher'
original=np.load(R/'clean_r3_80_review_trace.npz')
parts.append((original['obs'][150:,0],original['raw'][150:,0],12.))
X=np.concatenate([a for a,b,w in parts]).astype(np.float32);Y=np.concatenate([b for a,b,w in parts]).astype(np.float32)
weight=np.concatenate([np.full(len(a),w) for a,b,w in parts])
with torch.no_grad():
 hidden=model.net[:-1]( (torch.tensor(X)-model.mean)/(model.std+.01) ).numpy()
 old=model(torch.tensor(X)).numpy()
delta=Y[:,12]-old[:,12]
assert np.max(np.abs((Y[:,11]-old[:,11])-.74*delta))<1e-4
assert np.max(np.abs((Y[:,13]-old[:,13])-.26*delta))<1e-4
H=np.c_[hidden,np.ones(len(hidden))].astype(np.float64)
A=H.T@(weight[:,None]*H);b=H.T@(weight*delta)
records=[]
for ridge in [.2,2.,20.,200.]:
 penalty=np.eye(H.shape[1])*ridge;penalty[-1,-1]=ridge*.1
 coeff=np.linalg.solve(A+penalty,b)
 ck=copy.deepcopy(parent)
 for row,gain in [(11,.74),(12,1.),(13,.26)]:
  ck['actor_state_dict']['mlp.6.weight'][row]+=torch.tensor(gain*coeff[:-1],dtype=torch.float32)
  ck['actor_state_dict']['mlp.6.bias'][row]+=float(gain*coeff[-1])
 for k,v in sd.items():
  if k not in ['mlp.6.weight','mlp.6.bias']:assert torch.equal(v,ck['actor_state_dict'][k])
 keep=[i for i in range(14) if i not in [11,12,13]]
 assert torch.equal(sd['mlp.6.weight'][keep],ck['actor_state_dict']['mlp.6.weight'][keep])
 assert torch.equal(sd['mlp.6.bias'][keep],ck['actor_state_dict']['mlp.6.bias'][keep])
 name='startup_distilled_'+str(ridge).replace('.','p')
 ck.pop('optimizer_state_dict',None)
 ck.setdefault('infos',{})['startup_distillation']=dict(parent='clean r3 model80',ridge=ridge,changed_output_rows=[11,12,13],teacher_uses_actual_imu=True,actor_runtime_helper=False)
 torch.save(ck,R/(name+'.pt'))
 sd=ck['actor_state_dict'];export=Actor().eval()
 torch.onnx.export(export,torch.zeros(1,61),str(R/(name+'.onnx')),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
 ix=np.random.default_rng(1230091).choice(len(X),128,replace=False)
 with torch.no_grad():expected=export(torch.tensor(X[ix])).numpy()
 actual=np.concatenate([actor(name+'.onnx').run(None,{'obs':o[None]})[0] for o in X[ix]]);error=float(abs(actual-expected).max());assert error<2e-5
 prediction=H@coeff
 records.append(dict(name=name,ridge=ridge,onnx_error=error,weighted_rmse_rad=float(np.sqrt(np.average((prediction-delta)**2,weights=weight))*.12),steady_correction_rmse_rad=float(np.sqrt(np.mean(prediction[-450:]**2))*.12)))
 sd=parent['actor_state_dict']
(R/'startup_distillation_manifest.json').write_text(json.dumps(dict(selected=selected,frames=len(X),source_checkpoint='ppo_clean_r3_run/model_80.pt',actor_dim=61,action_dim=14,changed_rows=[11,12,13],all_other_actor_weights_unchanged=True,normalizer_std_critic_unchanged=True,training='coupled output-layer weighted ridge regression',records=records),indent=2))
print('DISTILLED',records,flush=True)
