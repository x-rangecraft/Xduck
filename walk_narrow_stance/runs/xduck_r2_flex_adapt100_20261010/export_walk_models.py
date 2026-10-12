from common import *
import argparse,torch,json,hashlib

ap=argparse.ArgumentParser();ap.add_argument('iterations',nargs='+',type=int)
ap.add_argument('--run_dir',default='ppo_run');ap.add_argument('--prefix',default='walk_ppo');args=ap.parse_args()
parent=torch.load(R/'clean_parent.pt',map_location='cpu',weights_only=False)
class Actor(torch.nn.Module):
    def __init__(self,sd):
        super().__init__()
        self.register_buffer('mean',sd['obs_normalizer._mean'].clone())
        self.register_buffer('std',sd['obs_normalizer._std'].clone())
        self.net=torch.nn.Sequential(torch.nn.Linear(61,512),torch.nn.ELU(),
            torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,14))
        self.net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('mlp.')})
    def forward(self,x):
        return self.net((x-self.mean)/(self.std+.01))
records=[]
for iteration in args.iterations:
    path=R/args.run_dir/f'model_{iteration}.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False);sd=ck['actor_state_dict']
    for k,v in sd.items():
        assert torch.isfinite(v).all()
        if k.startswith('obs_normalizer.'):
            assert torch.equal(v,parent['actor_state_dict'][k]),k
    model=Actor(sd).eval()
    name=f'{args.prefix}_{iteration}.onnx'
    torch.onnx.export(model,torch.zeros(1,61),str(R/name),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
    sample=np.random.default_rng(1160091).normal(0,.2,(128,61)).astype(np.float32)
    with torch.no_grad():expected=model(torch.tensor(sample)).numpy()
    session=actor(name);actual=np.concatenate([session.run(None,{'obs':o[None]})[0] for o in sample])
    error=float(abs(actual-expected).max());assert error<2e-5
    records.append(dict(iteration=iteration,onnx=name,error=error,actor_normalizer_frozen=True,
                        completed_updates=ck['infos']['unilab_progress']['next_iteration'],
                        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(R/(args.prefix+'_export_validation.json')).write_text(json.dumps(records,indent=2));print('EXPORTED',records,flush=True)
