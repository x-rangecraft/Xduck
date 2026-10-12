from common import *
import torch,json,hashlib
p=R/'clean_parent.pt'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='04595e1eb5736b6cac43f25613711156ed4ff02daf78684c860156aba3bbce41'
ck=torch.load(p,map_location='cpu',weights_only=False);sd=ck['actor_state_dict']
class Actor(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',sd['obs_normalizer._mean'].clone());self.register_buffer('std',sd['obs_normalizer._std'].clone())
  self.net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14))
  self.net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')})
 def forward(self,x):return self.net((x-self.mean)/(self.std+.01))
m=Actor().eval();torch.onnx.export(m,torch.zeros(1,61),str(R/'clean_parent.onnx'),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
x=np.random.default_rng(1190091).normal(0,.2,(128,61)).astype(np.float32)
with torch.no_grad():expected=m(torch.tensor(x)).numpy()
a=actor('clean_parent.onnx');actual=np.concatenate([a.run(None,{'obs':o[None]})[0] for o in x]);error=float(abs(actual-expected).max());assert error<2e-5
(R/'clean_export_validation.json').write_text(json.dumps(dict(parent_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),onnx_error=error,actor_dim=61,action_dim=14,weights_changed=False),indent=2));print('CLEAN_PARENT_VERIFIED',error,flush=True)
