from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parent
s=json.loads((R/'status.json').read_text())
assert not s.get('user_requires_explicit_resume',True),'Explicit start authorization required'
assert not list(R.glob('block_*_completion.json')),'Refusing to overwrite a previous run'
for name in ['reward_verification.json','environment_verification.json']:assert json.loads((R/name).read_text())['passed']
smoke=json.loads((R/'smoke_v2_completion.json').read_text());assert smoke['initialized_from_scratch'] and smoke['source_checkpoint'] is None
with (R/'driver.log').open('w') as f:p=subprocess.Popen([sys.executable,str(R/'run_job.py')],cwd=R/'project',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'driver_pid.json').write_text(json.dumps({'pid':p.pid,'fresh_initialization':True,'reward_version':'tilt_brake_v2'},indent=2)+'\n');print(p.pid)
