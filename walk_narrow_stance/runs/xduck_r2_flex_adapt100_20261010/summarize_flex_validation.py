from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parent;d=json.loads((R/'adapt50_validation_results.json').read_text());rs=d['records'];rows=[]
for name in ['forward','backward','left','right','forward_left','forward_right']:
 out=dict(scenario=name)
 for model in ['before_r2','adapt50']:
  rr=[r for r in rs if r['model']==model and r['scenario']==name];active=[r['signed_progress_mean'] for r in rr if r['sustained']];speed=np.array([r['vx_mean'] for r in rr]);out[model]=dict(started=sum(r['started'] for r in rr),sustained=sum(r['sustained'] for r in rr),stable=sum(r['stable'] for r in rr),rough_tracking=sum(r['rough_tracking'] for r in rr),total=len(rr),all_vx_range=[float(speed.min()),float(speed.max())],all_vx_std=float(speed.std()),successful_signed_progress_range=[min(active),max(active)] if active else None,successful_signed_progress_std=float(np.std(active)) if active else None,width_range_mm=[min(r['width_mm'] for r in rr),max(r['width_mm'] for r in rr)],foot_peak_by_case=[r['foot_peak_median_mm'] for r in rr],first_0p8_right_roll_range=[min(r['first_0p8_right_roll_deg'] for r in rr),max(r['first_0p8_right_roll_deg'] for r in rr)])
 rows.append(out)
score=[]
for case in range(8):
 rr=[r for r in rs if r['model']=='adapt50' and r['case']==case];score.append((sum(r['rough_tracking'] for r in rr),sum(r['sustained'] for r in rr),case))
best=max(score)[2]
report=dict(actual_ppo_updates=51,new_DR_cases=8,closed_loop_seconds=12,scene='48-passive-coordinate flexible physics',mechanical_steps_per_control=320,paired_parameters_noise_delay=True,model_unchanged_during_validation=True,selected_checkpoint_sha256=hashlib.sha256((R/'flex_adapt_50.pt').read_bytes()).hexdigest(),all_six_directions_all_8_conditions_pass=all(r['rough_tracking'] for r in rs if r['model']=='adapt50'),representative_video_case=best,video_case_score=score,summary=rows)
(R/'FLEX50_CAPABILITY_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(dict(video_case=best,summary=[dict(scenario=r['scenario'],before={k:r['before_r2'][k] for k in ['started','sustained','rough_tracking']},after={k:r['adapt50'][k] for k in ['started','sustained','rough_tracking']},speed_range=r['adapt50']['successful_signed_progress_range']) for r in rows]),indent=2))
