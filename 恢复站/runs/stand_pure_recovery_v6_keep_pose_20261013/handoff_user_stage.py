from pathlib import Path
import json,os,signal,subprocess
R=Path(__file__).resolve().parent;s=json.loads((R/'status.json').read_text());assert s['phase']=='training' and s['completed_updates']==1400 and s['block']=='block_14'
needle=f'{R}/project/.venv/bin/python {R}/train.py --run {s["block"]} '
ids=[int(x.split()[0]) for x in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if len(x.split(None,1))==2 and x.split(None,1)[1].startswith(needle)];assert len(ids)==1
pid=json.loads((R/'driver_pid.json').read_text())['pid'];assert str(R/'run_job.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
os.kill(pid,signal.SIGTERM)
request=dict(after_total_updates=1800,target_level=3,reset_push_stage=0,require_training_and_evaluation_complete=True,user_authorized=True,applied=False)
(R/'USER_STAGE_TRANSITION.json').write_text(json.dumps(request,indent=2)+'\n')
s.update(phase='finishing_active_block',training_running=True,pending_block=s['block'],pending_trainer_pid=ids[0],next_course_level=3,user_authorized_transition_after_total_updates=1800)
tmp=R/'status_update.tmp';tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(R/'status.json');print(json.dumps(dict(active_trainer_preserved=ids[0],request=request)),flush=True)
