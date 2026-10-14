from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parent;rows=[]
for model,checkpoint in [(800,'block_04/final.pt'),(1200,'block_08/final.pt')]:
 for force,duration in [(18,.4),(24,.4),(30,.4),(18,.6),(24,.6),(30,.6),(18,1.),(24,1.),(30,1.)]:
  tag=f'diag_{model}_{force}N_{int(duration*1000)}ms';out=R/(tag+'_results.json')
  if not out.exists():
   with (R/(tag+'.log')).open('w') as f:subprocess.run([str(Path.home()/'.local/bin/uv'),'run','--no-project','--python',sys.executable,str(R/'diagnose_push_case.py'),'--checkpoint',checkpoint,'--tag',tag,'--phase','settle','--challenge-level','2','--force-N',str(force),'--force-duration',str(duration),'--single-push'],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R/'project')
  result=json.loads(out.read_text());push=[x for x in result['rows'] if x['name']=='course_push']
  row=dict(model=model,force_N=force,duration_s=duration,impulse_Ns=force*duration,nonfall=sum(not x['failed'] for x in push),cases=len(push),surviving_directions=[x['case']-24 for x in push if not x['failed']],initial_signature=result['push_diagnostic']['initial_signature'])
  rows.append(row);(R/'PUSH_DIFFICULTY_GRID.json').write_text(json.dumps(dict(complete=False,rows=rows),indent=2)+'\n');print(json.dumps(row),flush=True)
assert len({x['initial_signature'] for x in rows})==1
(R/'PUSH_DIFFICULTY_GRID.json').write_text(json.dumps(dict(complete=True,same_initialization_verified=True,DR_level=2,DR_multiplier=2.,single_push=True,rows=rows),indent=2)+'\n')
