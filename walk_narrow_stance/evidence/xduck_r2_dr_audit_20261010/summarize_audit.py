from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent;names=['forward','backward','left','right','forward_left','forward_right'];results=[]
for name in names:
 d=json.loads((R/('training_sensors_symmetric_'+name+'_results.json')).read_text());rs=d['records'];groups={}
 for group in ['all','random','endpoint','nominal']:
  rr=rs if group=='all' else [r for r in rs if r['group']==group];active=np.array([r['signed_speed_or_turn_rate'] for r in rr if r['sustained_12s']]);allvalues=np.array([r['signed_speed_or_turn_rate'] for r in rr]);mean=float(active.mean()) if len(active) else None;std=float(active.std()) if len(active) else None
  groups[group]=dict(total=len(rr),started_by_2p5s=sum(r['started_by_2p5s'] for r in rr),sustained_12s_gate=sum(r['sustained_12s'] for r in rr),stable_all=sum(r['stable_all'] for r in rr),within_median_20pct_including_failures=sum(r['speed_within_median_20pct'] for r in rr),active_speed_range=[float(active.min()),float(active.max())] if len(active) else None,all_speed_range=[float(allvalues.min()),float(allvalues.max())],active_mean=mean,active_std=std,active_cv=std/mean if mean else None,active_speed_p05_p95=np.percentile(active,[5,95]).tolist() if len(active) else None)
 results.append(dict(command=name,unit='rad/s' if name in ['left','right'] else 'm/s',active_global_median=d['active_signed_speed_median'],groups=groups,failures=[r for r in rs if not r['sustained_12s']]))
report=dict(conclusion_all_dr_start_and_same_speed=False,not_a_continuous_parameter_exhaustion=True,conditions_per_command=159,total_six_command_cases=954,random_conditions=128,endpoint_conditions=30,nominal_conditions=1,standing_pose='symmetric',model='direction_r2_179.onnx',uses_actual_training_uniform_noise_and_imu_delay=True,standing599_calls=0,simulation_only=True,definition='start within 2.5s: both feet actual >12mm swings (>8mm for pure turns), correct signed progress and stable; sustained: timely start + balance throughout + >=6 proper landings per foot after 3s + positive 1s progress windows; speed within successful-condition median ±20%, failed cases count against passing',results=results)
(R/'DR_AUDIT_SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for r in results:
 g=r['groups']['all'];print(r['command'],g['started_by_2p5s'],g['sustained_12s_gate'],g['active_speed_range'],round(g['active_cv'],3),g['within_median_20pct_including_failures'])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
for ax,name in zip(axs,['forward','backward']):
 d=json.loads((R/('training_sensors_symmetric_'+name+'_results.json')).read_text());rs=d['records'];v=np.array([r['signed_speed_or_turn_rate'] for r in rs]);ok=np.array([r['sustained_12s'] for r in rs]);med=d['active_signed_speed_median'];i=np.arange(len(rs));ax.axhspan(.8*med,1.2*med,color='#80bc91',alpha=.25,label='successful-case median +/-20%');ax.axhline(.195,c='#697785',linestyle='--',label='training speed target');ax.axvline(30.5,c='#a0a0a0',linestyle=':');ax.scatter(i[ok],v[ok],s=14,c='#3077bd',label='passed start + sustained gate');ax.scatter(i[~ok],v[~ok],s=32,c='#d94646',marker='x',label='failed gate');ax.set_ylabel(name+' signed speed (m/s)');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left',ncol=2)
axs[1].set_xlabel('DR condition: 0 nominal | 1-30 endpoints | 31-158 new random combinations');fig.suptitle('Unmodified R2 PPO mean actor | symmetric standing pose | full training sensor noise/delay');fig.tight_layout();fig.savefig(R/'dr_speed_distribution.png',dpi=150);plt.close(fig)
