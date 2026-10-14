from pathlib import Path
import json,subprocess,sys
R=Path(__file__).resolve().parent
if (R/'driver_pid.json').exists():
 pid=json.loads((R/'driver_pid.json').read_text())['pid'];p=Path(f'/proc/{pid}/cmdline');assert not p.exists() or str(R/'run_job.py').encode() not in p.read_bytes()
with (R/'driver.log').open('w') as f:process=subprocess.Popen([sys.executable,str(R/'run_job.py')],cwd=R/'project',stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(R/'driver_pid.json').write_text(json.dumps(dict(pid=process.pid),indent=2)+'\n');print(process.pid,flush=True)
