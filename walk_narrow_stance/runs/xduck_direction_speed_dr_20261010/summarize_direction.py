from pathlib import Path
import argparse,json,numpy as np
p=argparse.ArgumentParser();p.add_argument('results',nargs='+');a=p.parse_args();output=[]
for name in a.results:
 d=json.loads(Path(name).read_text());rs=d['records'];rows=[]
 for kind in dict.fromkeys(r['scenario'] for r in rs):
  rr=[r for r in rs if r['scenario']==kind];v=np.array([r['vx_mean_mps'] for r in rr]);target=rr[0]['speed_target_mps'];ok=[r for r in rr if r['rough_tracking_pass']];active=np.array([r['vx_mean_mps'] for r in rr if r['stable'] and r['translation_direction_ok']]) if target else np.array([])
  row=dict(command=kind,pass_count=len(ok),total=len(rr),stable=sum(r['stable'] for r in rr),all_velocity_range=[float(v.min()),float(v.max())],active_velocity_std=float(active.std()) if len(active)>1 else None,active_velocity_count=len(active),speed_target=target)
  if 'steady_width_mm' in rr[0]:row.update(width_range=[min(r['steady_width_mm'] for r in rr),max(r['steady_width_mm'] for r in rr)],foot_peak_by_case=[r['side_peak_median_mm'] for r in rr])
  rows.append(row)
 out=dict(file=name,model=d['model'],parameter_bank=d['parameter_bank'],summary=rows);output.append(out);print(json.dumps(out,ensure_ascii=False,indent=2))
Path('direction_validation_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
