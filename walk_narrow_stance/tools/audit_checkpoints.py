"""Read metadata from the 11 user-owned checkpoints; never executes training."""
from pathlib import Path
import json,hashlib,torch
R=Path(__file__).resolve().parents[1];torch.set_num_threads(2);sources=json.loads((R/'checkpoint_sources.json').read_text());out=[]
for source in sources:
 p=R/'checkpoints'/source['file'];actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==source['sha256']
 ck=torch.load(p,map_location='cpu',weights_only=False);sd=ck.get('actor_state_dict',{});progress=ck.get('infos',{}).get('unilab_progress',{});dims={k:list(v.shape) for k,v in sd.items() if isinstance(v,torch.Tensor) and k in ['mlp.0.weight','mlp.6.weight']};norm={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in sd.items() if k.startswith('obs_normalizer.') and isinstance(v,torch.Tensor)}
 std=sd.get('distribution.std_param');std_range=[float(std.min()),float(std.max())] if std is not None else None
 out.append(dict(file=source['file'],sha256=actual,stored_iteration=ck.get('iter'),progress=progress,actor_weight_shapes=dims,normalizer_buffers=norm,exploration_std_range=std_range))
(R/'CHECKPOINT_METADATA.json').write_text(json.dumps(out,indent=2));print('CHECKPOINTS_VERIFIED',len(out))
