from pathlib import Path
R=Path(__file__).resolve().parent;p=R/'project/src/unilab/envs/locomotion/xduck_gf43x40/rigid_gpu.py';s=p.read_text();s=s.replace('        self._nominal_geom_pos=bm.geom_pos.copy();', '        self._nominal_flex_damping=bm.dof_damping.copy()\n        self._nominal_geom_pos=bm.geom_pos.copy();');s=s.replace('            d[:,env._motor_vel_indices]=0.\n','');p.write_text(s)
p=R/'test_flex.py';s=p.read_text().replace('d[:,e._motor_vel_indices]','d[:,m.jnt_dofadr[m.actuator_trnid[:,0]]]');p.write_text(s)
p=R/'supervisor.py';s=p.read_text().replace("(R/(tag+'.log')).open('x')", "(R/(tag+'.log')).open('a')");p.write_text(s)
