from pathlib import Path
import json,os,signal,shutil
R=Path(__file__).resolve().parent
s=json.loads((R/'status.json').read_text());assert s['completed_updates']==200 and s['block']=='block_02'
trainer=2801400;assert str(R/'train.py').encode() in Path(f'/proc/{trainer}/cmdline').read_bytes()
pid=json.loads((R/'driver_pid.json').read_text())['pid'];assert str(R/'run_job.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
os.kill(pid,signal.SIGTERM)
for i,tags in enumerate([('block_00','block_00_dr_course'),('block_01','block_01_course')]):
 n,c=[json.loads((R/(tag+'_window_v2_results.json')).read_text()) for tag in tags]
 row=s['history'][i];row['previous_validation_accepted']=row['accepted'];row.update(accepted=True,validation_criteria_version='window_v2',nominal=n['summary']['model'],course=c['summary']['model'],overall_success_rate=(n['summary']['model']['settled']+c['summary']['model']['settled'])/64)
s.update(phase='finishing_active_block',training_running=True,best_checkpoint='block_01/final.pt',latest_checkpoint='block_01/final.pt',pending_block='block_02',pending_trainer_pid=trainer,next_course_level=1,validation_criteria_version='window_v2',consecutive_rejections=0)
shutil.copy2(R/'block_01/final.pt',R/'best.pt');tmp=R/'status_update.tmp';tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(R/'status.json')
print('Old supervisor stopped; active trainer retained. Next complete block will use level1 / DR1.5.',flush=True)
