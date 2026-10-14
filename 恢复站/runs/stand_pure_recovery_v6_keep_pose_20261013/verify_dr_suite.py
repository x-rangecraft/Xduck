from pathlib import Path
import json
from dr_suite_gate import passes,overall_success
R=Path(__file__).resolve().parent
c=json.loads((R/'model800_course_dr_only_results.json').read_text());w=json.loads((R/'model800_walk_stop_results.json').read_text())
assert c['domain_randomization_enabled'] and w['effective_dr']['enabled']
assert c['summary']['model']['cases']==32 and sum(x['cases'] for x in c['groups'].values())==32
assert w['summary']['attempted']==48 and w['no_reset_at_switch'] and w['continuous_physics_and_motor_history']
assert not w['used_for_training'] and not w['teacher_or_trajectory_training']
assert all(x['first_target_jump_rad']<=.110001 for x in w['rows'])
assert w['walker_sha256']=='e959c81f9a362bc6c3c0a4f2cc626716c41d54fafad2ff74debe900e65458ad9'
assert overall_success(c,w)==76/80 and passes(c,w)
bad=json.loads(json.dumps(w));bad['summary']['successful_stops']=40
assert not passes(c,bad)
no_dr=json.loads(json.dumps(c));no_dr['domain_randomization_enabled']=False
try:passes(no_dr,w)
except AssertionError:pass
else:raise AssertionError('No-DR evaluation must not enter the upgrade gate')
result=dict(passed=True,evaluation_scope='current-level DR only',groups=['quiet','tilt_and_velocity_recovery','physical_push','real_walk_to_stand'],nominal_evaluation_disabled=True,walker_used_for_training=False,source_walker_sha256=w['walker_sha256'],model800_course=c['groups'],model800_walk_stop=w['summary'],overall_success_rate=overall_success(c,w),threshold=.9,invalid_walking_attempts_count_in_overall_denominator=True)
(R/'DR_SUITE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
