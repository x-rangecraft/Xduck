from pathlib import Path
import json,subprocess,sys
R=Path(__file__).resolve().parent
assert json.loads((R/'FINAL_GOAL_VERIFICATION.json').read_text())['passed']
assert json.loads((R/'REWARD_GATE_VERIFICATION.json').read_text())['passed']
assert json.loads((R/'smoke_completion.json').read_text())['actor_weight_max_change']>0
assert not json.loads((R/'status.json').read_text()).get('user_requires_explicit_resume',False)
if (R/'driver_pid.json').exists():
 pid=json.loads((R/'driver_pid.json').read_text())['pid'];p=Path(f'/proc/{pid}/cmdline');assert not p.exists() or str(R/'run_job.py').encode() not in p.read_bytes()
with (R/'driver.log').open('w') as f:p=subprocess.Popen([sys.executable,str(R/'run_job.py')],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(R/'driver_pid.json').write_text(json.dumps(dict(pid=p.pid),indent=2)+'\n');print(p.pid,flush=True)
