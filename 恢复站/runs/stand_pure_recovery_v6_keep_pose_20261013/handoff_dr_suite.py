from pathlib import Path
import json,os,signal,subprocess
R=Path(__file__).resolve().parent
assert json.loads((R/'DR_SUITE_VERIFICATION.json').read_text())['passed']
s=json.loads((R/'status.json').read_text());assert s['phase']=='training'
block=s['block'];needle=f'{R}/project/.venv/bin/python {R}/train.py --run {block} '
lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();ids=[int(x.split()[0]) for x in lines if len(x.split(None,1))==2 and x.split(None,1)[1].startswith(needle)]
assert len(ids)==1,ids;trainer=ids[0]
pid=json.loads((R/'driver_pid.json').read_text())['pid'];assert str(R/'run_job.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
os.kill(pid,signal.SIGTERM)
s.update(phase='finishing_active_block',training_running=True,pending_block=block,pending_trainer_pid=trainer,next_course_level=s['level'],evaluation_scope='current_course_DR_and_real_walk_stop_only')
if s['completed_updates']>=500:
 c=json.loads((R/'model800_course_dr_only_results.json').read_text());w=json.loads((R/'model800_walk_stop_results.json').read_text())
 row=s['history'][4];row['previous_upgrade_summary']={k:row[k] for k in ['nominal','course','overall_success_rate'] if k in row}
 row.update(dr_only=True,course=c['summary']['model'],groups=c['groups'],walk_stop=w['summary'],overall_success_rate=.95,accepted=True)
tmp=R/'status_update.tmp';tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(R/'status.json')
manifest=json.loads((R/'CHANGE_MANIFEST.json').read_text());manifest['evaluation_scope_revision']=dict(no_DR_tests_removed=True,quiet_cases=4,tilt_and_velocity_cases=20,push_cases=8,real_walk_stop_attempts=48,dr_enabled_for_all=True,real_walking_checkpoint='walker_flex_adapt_50.pt',no_reset_at_handoff=True,walker_used_for_training=False,overall_gate_threshold=.9)
(R/'CHANGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(active_trainer_preserved=trainer,pending_block=block,level=s['level'],completed_additional_updates=s['completed_updates'],evaluation_scope=s['evaluation_scope'])),flush=True)
