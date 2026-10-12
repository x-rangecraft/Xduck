from common import *
import torch,json,hashlib,copy
parent=torch.load(R/'clean_parent.pt',map_location='cpu',weights_only=False)
z=np.load(R/'clean_baseline_trace.npz')
correction=np.array([.08,-.08])/D.WALK_ACTION_SCALE[[1,10]]-z['raw'][0][:,[1,10]].mean(0)
class Actor(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',sd['obs_normalizer._mean'].clone());self.register_buffer('std',sd['obs_normalizer._std'].clone())
  self.net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14))
  self.net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')})
 def forward(self,x):return self.net((x-self.mean)/(self.std+.01))
records=[]
for factor in [.6,.8,1.,1.2]:
 ck=copy.deepcopy(parent);sd=ck['actor_state_dict']
 sd['mlp.6.bias'][[1,10]]+=torch.tensor(correction*factor,dtype=sd['mlp.6.bias'].dtype)
 name='clean_bias_'+str(round(factor*100))
 torch.save(ck,R/(name+'.pt'));m=Actor().eval()
 torch.onnx.export(m,torch.zeros(1,61),str(R/(name+'.onnx')),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
 x=z['obs'][0].astype(np.float32)
 with torch.no_grad():expected=m(torch.tensor(x)).numpy()
 actual=np.concatenate([actor(name+'.onnx').run(None,{'obs':o[None]})[0] for o in x]);error=float(abs(actual-expected).max());assert error<2e-5
 records.append(dict(name=name,bias_correction=(correction*factor).tolist(),other_actor_weights_unchanged=True,onnx_error=error))
(R/'clean_bias_candidates.json').write_text(json.dumps(records,indent=2));print(records,flush=True)
