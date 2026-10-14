from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parent
assert json.loads((R/'PREFLIGHT.json').read_text())['passed'];assert not list(R.glob('block_*_completion.json'))
with (R/'driver.log').open('w') as f:p=subprocess.Popen([sys.executable,str(R/'run_job.py')],cwd=R/'project',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'driver_pid.json').write_text(json.dumps(dict(pid=p.pid,new_training_authorized=True),indent=2)+'\n');print(p.pid,flush=True)
