from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parent
assert json.loads((R/'REPAIR_VERIFICATION.json').read_text())['passed']
assert json.loads((R/'status.json').read_text())['user_requires_explicit_resume'] is False
if (R/'driver_pid.json').exists():
 pid=json.loads((R/'driver_pid.json').read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
 assert not cmd.exists() or str(R/'run_job.py').encode() not in cmd.read_bytes(),'A supervisor is already running'
with (R/'driver.log').open('w') as f:p=subprocess.Popen([sys.executable,str(R/'run_job.py')],cwd=R/'project',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'driver_pid.json').write_text(json.dumps(dict(pid=p.pid,new_training_authorized=True),indent=2)+'\n');print(p.pid,flush=True)
