from common import *
import torch,json
ck=torch.load(R/'ppo_clean_r3_run/model_100.pt',map_location='cpu',weights_only=False);sd=ck['actor_state_dict']
class Actor(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',sd['obs_normalizer._mean'].clone());self.register_buffer('std',sd['obs_normalizer._std'].clone())
  self.net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14))
  self.net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')})
 def forward(self,x):return self.net((x-self.mean)/(self.std+.01))
class DampedActor(Actor):
 def __init__(self,gain):super().__init__();self.gain=gain
 def forward(self,x):
  y=super().forward(x)
  delta=torch.clamp(self.gain*(x[:,2:3]-x[:,50:51]),-.06,.06)/.12
  left=.25*(y[:,0:1]+delta)+.75*x[:,34:35]
  right=.25*(y[:,9:10]+delta)+.75*x[:,43:44]
  return torch.cat([left,y[:,1:9],right,y[:,10:]],dim=1)
records=[]
for name,gain in [('clean_yaw_zero',0.),('clean_yaw_pos',.2),('clean_yaw_neg',-.2)]:
 m=DampedActor(gain).eval()
 torch.onnx.export(m,torch.zeros(1,61),str(R/(name+'.onnx')),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
 x=np.random.default_rng(1210091).normal(0,.2,(128,61)).astype(np.float32)
 with torch.no_grad():expected=m(torch.tensor(x)).numpy()
 actual=np.concatenate([actor(name+'.onnx').run(None,{'obs':o[None]})[0] for o in x]);error=float(abs(actual-expected).max());assert error<2e-5
 records.append(dict(name=name,gain=gain,filter_alpha=.25,angle_limit_rad=.06,actual_actor_inputs_only=True,actor_dim=61,action_dim=14,onnx_error=error,source='clean r3 model100',unmodified_joint_rows=[1,2,3,4,5,6,7,8,10,11,12,13]))
(R/'gyro_export_validation.json').write_text(json.dumps(records,indent=2));print(records,flush=True)
