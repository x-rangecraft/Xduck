from pathlib import Path
import json,shutil
from omegaconf import OmegaConf
from course_gate import passes,overall_success
R=Path(__file__).resolve().parent
assert json.loads((R/'DR_COURSE_VERIFICATION.json').read_text())['passed']
s=json.loads((R/'status.json').read_text());assert s['completed_updates']==100
n=json.loads((R/'block_00_results.json').read_text());d=json.loads((R/'block_00_dr_course_results.json').read_text())
assert d['domain_randomization_enabled'] and d['effective_dr']['multiplier']==1.
accepted=passes(n,d);rate=overall_success(n,d)
s['history'][0]['superseded_gate_accepted']=s['history'][0]['accepted'];s['history'][0].update(accepted=accepted,course_with_dr=d['summary']['model'],overall_success_rate=rate,reevaluated_under_user_rule=True)
s.update(phase='course_rule_updated',training_running=False,user_requires_explicit_resume=False,level=1 if accepted else 0,latest_checkpoint='block_00/final.pt',best_checkpoint='block_00/final.pt' if accepted else 'block_source/final.pt',pass_streak=0,consecutive_rejections=0,dr_multipliers=[1.,1.5,2.,2.5],upgrade_overall_success_threshold=.9)
(R/'status.json').write_text(json.dumps(s,indent=2)+'\n')
if accepted:shutil.copy2(R/'block_00/final.pt',R/'best.pt')
c=OmegaConf.load(R/'TRAINING_CONFIG.yaml');c.env.dr_curriculum_enabled=True;c.env.dr_multipliers=[1.,1.5,2.,2.5];c.env.discovery_level=s['level'];OmegaConf.save(c,R/'TRAINING_CONFIG.yaml')
manifest=json.loads((R/'CHANGE_MANIFEST.json').read_text())
manifest.update(course_changes=['DR multipliers 1,1.5,2,2.5 by levels 0..3','Scale trunk mass/inertia deviation, friction deviation from 1, armature deviation from 1, trunk/head COM and observation noise only','Keep sole installation error +-3deg; keep gains, motor ranges, IMU mounting and delays fixed','Push follows original force/duration course; no second multiplier','Overall nominal32 + current-course DR32 success >=90% upgrades immediately','Below threshold continue latest checkpoint at same level; preserve accepted checkpoint separately','Success remains nonfall and correct pose quiet during final continuous one second'],validation='Four instantiated physical environments verified effective DR ranges and finite steps',limitations='64 analytical evaluation episodes with nominal and sampled DR conditions; not qualification of actual walking/sitstand handoffs.')
(R/'CHANGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(overall_success_rate=rate,accepted=accepted,next_level=s['level'],next_dr_multiplier=[1.,1.5,2.,2.5][s['level']],nominal=n['summary']['model'],dr_course=d['summary']['model'])),flush=True)
