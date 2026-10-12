from pathlib import Path
import numpy as np,torch,json
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
sd=torch.load(R/'parent_r2.pt',map_location='cpu',weights_only=False)['actor_state_dict'];net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14));net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')});net.eval()
with np.load(R/'direction_anchor.npz') as z:data={k:z[k].copy() for k in z.files}
x=torch.tensor(data['obs'],dtype=torch.float32);pred=[]
with torch.inference_mode():
 for batch in x.split(512):pred.append(net((batch-sd['obs_normalizer._mean'])/(sd['obs_normalizer._std']+.01)).numpy())
data['mean']=np.concatenate(pred);np.savez_compressed(R/'direction_anchor.npz',**data)
(R/'preservation_relabel.json').write_text(json.dumps(dict(labels='selected R2 mean actor recomputed on preservation states',frames=len(x),phase_counts={str(i):int((data['phase']==i).sum()) for i in range(4)},old_teacher_action_labels_used=False),indent=2));print('R2_PRESERVATION_RELABELLED',len(x),flush=True)
