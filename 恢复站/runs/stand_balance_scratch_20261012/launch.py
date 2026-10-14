from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parent
if (R/'status.json').exists() and json.loads((R/'status.json').read_text()).get('phase')=='paused_by_user':
 raise SystemExit('User paused training; explicit resume instruction is required')
for name in ['external_force_verification.json','course_verification.json']:assert json.loads((R/name).read_text())['passed']
smoke=json.loads((R/'smoke_scratch_completion.json').read_text());assert smoke['initialized_from_scratch'] and smoke['mechanical_timestep_s']==.001
with (R/'driver.log').open('w') as f:p=subprocess.Popen([sys.executable,str(R/'run_job.py')],cwd=R/'project',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'driver_pid.json').write_text(json.dumps({'pid':p.pid},indent=2)+'\n');print(p.pid)
