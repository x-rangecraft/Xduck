from pathlib import Path
import json,signal,os,subprocess,shutil,hashlib
R=Path(__file__).resolve().parent;B=R.parent/'stand_pure_recovery_v6_keep_pose_20261013'
assert json.loads((R/'REPAIR_VERIFICATION.json').read_text())['passed']
old=json.loads((B/'status.json').read_text());pid=json.loads((B/'driver_pid.json').read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
if cmd.exists():assert str(B/'run_job.py').encode() in cmd.read_bytes();os.kill(pid,signal.SIGTERM)
s=json.loads((R/'status.json').read_text())
if old['phase']=='training':
 name=old['block'];needle=f'{B}/project/.venv/bin/python {B}/train.py --run {name} '
 ids=[int(x.split()[0]) for x in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if len(x.split(None,1))==2 and x.split(None,1)[1].startswith(needle)]
 assert len(ids)==1
 s['pending_source']=dict(completion_file=str(B/(name+'_completion.json')),trainer_pid=ids[0],previous_total_updates=old['lineage_compute_updates'])
 old.update(phase='finishing_old_version_for_repair',repair_destination=str(R),trainer_preserved=ids[0])
else:
 assert old['phase'] in ['waiting_for_gpu','block_review_complete','evaluating','curriculum_complete','failed']
 source=B/old['latest_checkpoint'];shutil.copy2(source,R/'block_source/final.pt');m=json.loads((R/'REPAIR_MANIFEST.json').read_text());m.update(source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_lineage_updates=old['lineage_compute_updates']);(R/'REPAIR_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
 old.update(phase='superseded_by_contact_repair',training_running=False,repair_destination=str(R))
(R/'status.json').write_text(json.dumps(s,indent=2)+'\n');(B/'status.json').write_text(json.dumps(old,indent=2)+'\n');print(json.dumps(dict(previous_phase=old['phase'],pending_source=s.get('pending_source'))),flush=True)
