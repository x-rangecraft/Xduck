from engine import *
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=1210091);parser.add_argument('--prefix',default='accepted_eval_params');args=parser.parse_args()
np.random.seed(args.seed);torch.manual_seed(args.seed)
e=env(24)
try:
 e.init_state();e.set_autoreset(False)
 phi=e._backend.get_physics_state()
 fields={k:getattr(e._backend._device_model,k).numpy()[:8].copy() for k in FIELDS}
 fields.update(qpos=phi[:8,1:22],qvel=phi[:8,22:42],encoder_bias=e._encoder_bias[:8].copy(),imu_rotation=e._imu_rotation[:8].copy(),sole_angles=e._sole_angles[:8].copy(),sole_delta_quat=e._sole_delta_quat[:8].copy())
 np.savez_compressed(R/(args.prefix+'.npz'),**fields)
 print('ACCEPTED_PARAMETER_BANK_READY',flush=True)
finally:e.close()
