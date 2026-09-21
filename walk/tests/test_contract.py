"""Selected policy I/O and motor transmission contracts on the new runtime."""

import numpy as np

from xduck_walk_latest.task import make_walk_env, walk_cfg


def test_checkpoint_field_order_and_unit_gain_transmission():
    cfg = walk_cfg()
    cfg.observations["policy"].terms["legacy_order"].params["noise_level"] = 0.0
    env = make_walk_env(cfg, num_envs=1)
    try:
        env.reset()
        raw_action = np.full((1, 14), 0.05, dtype=np.float32)
        state = env.step(raw_action)
        actor = state.obs["obs"]
        critic = state.obs["critic"]
        command = env.command_manager.get_command("walk")
        np.testing.assert_array_equal(actor[:, 34:48], raw_action)
        np.testing.assert_array_equal(actor[:, 48:61], command)
        np.testing.assert_array_equal(critic[:, 37:51], raw_action)
        np.testing.assert_array_equal(critic[:, 51:54], command[:, :3])
        np.testing.assert_array_equal(critic[:, 66:70], command[:, 3:7])
        np.testing.assert_array_equal(critic[:, 70:76], command[:, 7:13])
        term = env.action_manager.get_term("joint_pos")
        # The selected V1.1.2 MJCF uses kp=1, kv=0; q + torque is the
        # position-control input. MuJoCo's actuatorfrc sensor verifies this
        # transmission through the published backend sensor contract.
        sensor = env.scene.bind_sensor_data(tuple(
            f"torque_{name}" for name in (
                "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
                "left_knee", "left_ankle", "neck_pitch", "head_pitch",
                "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll",
                "right_hip_pitch", "right_knee", "right_ankle",
            )
        ))
        actual_force = sensor.read()
        assert actual_force.shape == (1, 14)
        np.testing.assert_allclose(actual_force, term.motor.torque, atol=1e-4, rtol=1e-4)
    finally:
        env.close()
