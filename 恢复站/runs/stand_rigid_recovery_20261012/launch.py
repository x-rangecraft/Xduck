from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parent
assert json.loads((R/'external_force_verification.json').read_text())['passed']
assert json.loads((R/'smoke_rigid_completion.json').read_text())['flexible_dr_enabled']==False
with (R/'stage_driver.log').open('w') as f:
 child=subprocess.Popen([sys.executable,str(R/'run_stage.py')],cwd=R/'project',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'driver_pid.json').write_text(json.dumps({'pid':child.pid,'first_stage_updates':50,'flexible_dofs':0},indent=2)+'\n');print(child.pid)
