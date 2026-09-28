"""Near-risk checks for the reversed axes and preserved GF43 motor control of V1.1.2."""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.actuators.gf43x import GF43X40_10
from unilab.base import registry
from unilab.envs.locomotion.microduck_enlarged_112.tasks import (
    ASSET_ROOT,
    JOINT_NAMES,
    MicroDuckEnlarged112WalkFlatEnv,
)
from unilab.utils.rotation import np_quat_mul


def make_env(task, num_envs=4, extra=None):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[4] / "conf/ppo"), version_base="1.3"
    ):
        cfg = compose(
            config_name="config_mlx", overrides=[f"task=microduck_enlarged_112_{task}_flat/mujoco"]
        )
    override = OmegaConf.to_container(cfg.env, resolve=True)
    override["reward_config"] = OmegaConf.to_container(cfg.reward, resolve=True)
    override.update(extra or {})
    registry.ensure_registries()
    env = registry.make(
        cfg.training.task_name, sim_backend="mujoco", num_envs=num_envs, env_cfg_override=override
    )
    env.set_autoreset(False)
    return env


def _neck_head_roll_contacts(model, data):
    pairs = []
    for contact in data.contact:
        names = {
            model.geom(int(contact.geom1)).name,
            model.geom(int(contact.geom2)).name,
        }
        if "neck_pitch_collision" in names and any(
            name.startswith("head_roll_collision_") for name in names
        ):
            pairs.append(names)
    return pairs


def test_head_roll_collision_decomposition_removes_false_contact_but_keeps_protection():
    model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "scene_flat.xml"))
    data = mujoco.MjData(model)
    geom_names = {model.geom(i).name for i in range(model.ngeom)}
    head_collision_names = {
        name for name in geom_names if name.startswith("head_roll_collision_")
    }

    assert len(head_collision_names) == 25
    assert "head_roll_collision" not in geom_names
    assert model.nq == 21 and model.nv == 20 and model.nu == 14
    assert model.body_mass.sum() == pytest.approx(8.8933)

    # Exact HOME-settled, formal-actor and r1-actor poses had 9.0--17.3 mm
    # source-triangle clearance but contacted the former single convex hull.
    clear_head_poses = (
        (-0.26301020, -0.34859541, -0.00000288, 0.00002670),
        (-0.26140174, -0.60156846, -0.26531211, -0.17365170),
        (-0.27323347, -0.72828871, -0.34285548, -0.17332128),
    )
    for pose in clear_head_poses:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.qpos[12:16] = pose
        mujoco.mj_forward(model, data)
        assert not _neck_head_roll_contacts(model, data)

    # A separately audited source-triangle intersection must still contact:
    # the correction is a collision decomposition, not a disabled pair.
    real_interference_pose = (0.27614640, 1.16972825, -1.31392606, -0.44352671)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qpos[12:16] = real_interference_pose
    mujoco.mj_forward(model, data)
    assert _neck_head_roll_contacts(model, data)


def test_walk_command_modes_are_exclusive_and_match_requested_mix():
    from unilab.envs.locomotion.microduck_enlarged_112.tasks import V112WalkReset

    p = [0.2, 0.3, 0.1, 0.1, 0.15, 0.15]
    low, high = [-0.2, -0.15, -0.6], [0.35, 0.15, 0.6]
    cfg = SimpleNamespace(
        command_mode_probabilities=p,
        remote_command_probabilities=(),
        commands=SimpleNamespace(vel_limit=[low, high]),
    )
    np.random.seed(113)
    commands = V112WalkReset()._sample_commands(SimpleNamespace(cfg=cfg), 30000)
    active = commands != 0
    one = active.sum(axis=1) == 1
    modes = [
        ~active.any(axis=1),
        one & (commands[:, 0] > 0),
        one & (commands[:, 0] < 0),
        one & active[:, 1],
        one & active[:, 2],
        active.all(axis=1),
    ]
    np.testing.assert_array_equal(np.sum(modes, axis=0), 1)
    np.testing.assert_allclose(np.mean(modes, axis=1), p, atol=0.008)
    assert np.all(commands >= low) and np.all(commands <= high)
    assert np.any(commands[modes[3], 1] > 0) and np.any(commands[modes[3], 1] < 0)
    assert np.any(commands[modes[4], 2] > 0) and np.any(commands[modes[4], 2] < 0)


def test_remote_command_modes_never_emit_strafe_or_pure_yaw():
    from unilab.envs.locomotion.microduck_enlarged_112.tasks import V112WalkReset

    p = [0.2, 0.25, 0.15, 0.25, 0.15]
    cfg = SimpleNamespace(
        remote_command_probabilities=p,
        command_mode_probabilities=(),
        commands=SimpleNamespace(vel_limit=[[-0.15, 0.0, -0.5], [0.3, 0.0, 0.5]]),
    )
    np.random.seed(3311)
    commands = V112WalkReset()._sample_commands(SimpleNamespace(cfg=cfg), 30000)
    stand = np.all(commands == 0, axis=1)
    moving = ~stand
    arc = commands[:, 2] != 0

    assert np.all(commands[:, 1] == 0)
    assert np.all(np.abs(commands[moving, 0]) >= 0.08)
    assert np.all(commands[arc, 0] != 0)
    assert np.any(commands[arc, 0] > 0) and np.any(commands[arc, 0] < 0)
    assert np.any(commands[arc, 2] > 0) and np.any(commands[arc, 2] < 0)
    assert np.mean(stand) == pytest.approx(p[0], abs=0.008)


def test_remote_profile_changes_command_curriculum_and_calibrated_learning_rate_only():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[4] / "conf/ppo"), version_base="1.3"
    ):
        baseline = OmegaConf.to_container(
            compose(
                config_name="config_mlx",
                overrides=["task=microduck_enlarged_112_walk_flat/mujoco_limits"],
            ),
            resolve=True,
        )
        remote = OmegaConf.to_container(
            compose(
                config_name="config_mlx",
                overrides=["task=microduck_enlarged_112_walk_flat/mujoco_remote"],
            ),
            resolve=True,
        )

    assert remote["env"]["command_mode_probabilities"] == []
    assert remote["env"]["remote_command_probabilities"] == [0.2, 0.25, 0.15, 0.25, 0.15]
    assert remote["env"]["commands"]["vel_limit"] == [[-0.15, 0.0, -0.5], [0.3, 0.0, 0.5]]
    assert remote["algo"]["algorithm"]["learning_rate"] == 1.0e-5
    remote["algo"]["algorithm"]["learning_rate"] = baseline["algo"]["algorithm"][
        "learning_rate"
    ]
    for profile in (baseline, remote):
        profile["env"].pop("command_mode_probabilities", None)
        profile["env"].pop("remote_command_probabilities", None)
        profile["env"].pop("command_resample_interval_s", None)
        profile["env"]["commands"].pop("vel_limit", None)
        for section in ("training", "algo"):
            for field in ("seed", "log_root", "resume", "resume_trainer_state", "load_run"):
                profile[section].pop(field, None)
        profile["algo"]["max_iterations"] = 501
    assert remote == baseline


def test_head_bias_profile_adds_only_official_persistent_pose_objective():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[4] / "conf/ppo"), version_base="1.3"
    ):
        remote = OmegaConf.to_container(
            compose(
                config_name="config_mlx",
                overrides=["task=microduck_enlarged_112_walk_flat/mujoco_remote"],
            ),
            resolve=True,
        )
        candidate = OmegaConf.to_container(
            compose(
                config_name="config_mlx",
                overrides=["task=microduck_enlarged_112_walk_flat/mujoco_remote_head_bias"],
            ),
            resolve=True,
        )

    assert candidate["reward"]["head_pose_bias_tau_s"] == 1.0
    assert candidate["reward"]["head_pose_bias_schedule_steps"] == [0, 1200, 2400, 3600]
    assert candidate["reward"]["head_pose_bias_schedule_scales"] == [0.0, 1.0, 2.0, 3.0]
    assert candidate["reward"]["scales"].pop("head_pose_bias") == 0.0
    for section in ("training", "algo"):
        for field in ("seed", "log_root", "load_run", "max_iterations"):
            remote[section].pop(field, None)
            candidate[section].pop(field, None)
    for field in (
        "head_pose_bias_tau_s",
        "head_pose_bias_schedule_steps",
        "head_pose_bias_schedule_scales",
    ):
        candidate["reward"].pop(field)
    assert candidate == remote


def test_head_pose_bias_penalizes_dc_error_and_clears_on_reset():
    env = make_env("walk", 2)
    try:
        error = np.deg2rad(
            np.asarray([[0.0, 0.0, 10.0, -10.0], [5.0, -5.0, 0.0, 0.0]], dtype=np.float32)
        )
        info = {
            "head_commands": np.zeros((2, 4), dtype=np.float32),
            "head_pose_bias_ema": np.zeros((2, 4), dtype=np.float32),
            "steps": np.ones(2, dtype=np.int64),
        }
        ctx = SimpleNamespace(
            dof_pos=np.broadcast_to(env.default_angles, (2, 14)).copy(),
            default_angles=env.default_angles,
            info=info,
        )
        ctx.dof_pos[:, [5, 6, 7, 8]] += error
        alpha = env.cfg.ctrl_dt / env._reward_cfg.head_pose_bias_tau_s

        first = env._head_pose_bias_penalty(ctx)
        np.testing.assert_allclose(first, -np.mean(np.abs(alpha * error), axis=1), rtol=1e-6)
        second = env._head_pose_bias_penalty(ctx)
        expected_ema = ((1.0 - alpha) * alpha + alpha) * error
        np.testing.assert_allclose(second, -np.mean(np.abs(expected_ema), axis=1), rtol=1e-6)

        info["steps"][:] = 0
        reset = env._head_pose_bias_penalty(ctx)
        np.testing.assert_allclose(reset, first, rtol=1e-6)
    finally:
        env.close()


def test_head_level_bias_targets_head_shell_roll_not_four_joint_home():
    angle = np.deg2rad(20.0)
    q_home = np.asarray([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)

    def axis_quat(axis, value):
        vector = np.zeros(3, dtype=np.float32)
        vector[axis] = np.sin(value / 2)
        return np.concatenate(([np.cos(value / 2)], vector)).astype(np.float32)

    quats = np.stack(
        [
            q_home,
            np_quat_mul(axis_quat(0, angle), q_home),
            np_quat_mul(axis_quat(0, -angle), q_home),
            np_quat_mul(axis_quat(1, angle), q_home),
            np_quat_mul(axis_quat(2, angle), q_home),
        ]
    )

    class Backend:
        def get_body_quat_b(self, _ids):
            return quats[:, None, :]

    owner = SimpleNamespace(
        _backend=Backend(),
        _head_roll_body_ids=np.asarray([0]),
        _head_level_tau=1.0,
        _head_level_moving_multiplier=2.0,
        _gait_active=lambda info: np.asarray(info["moving"]),
        _cfg=SimpleNamespace(ctrl_dt=0.02),
    )
    info = {
        "steps": np.ones(5, dtype=np.int64),
        "moving": np.asarray([False, False, False, False, False]),
    }
    result = MicroDuckEnlarged112WalkFlatEnv._head_level_bias_penalty(
        owner, SimpleNamespace(info=info)
    )
    assert result[0] == pytest.approx(0.0, abs=1e-7)
    assert result[1] == pytest.approx(result[2], abs=1e-7)
    assert result[1] < 0
    assert result[3] == pytest.approx(0.0, abs=1e-7)
    assert result[4] == pytest.approx(0.0, abs=1e-7)

    info["steps"][:] = 0
    reset = MicroDuckEnlarged112WalkFlatEnv._head_level_bias_penalty(
        owner, SimpleNamespace(info=info)
    )
    np.testing.assert_allclose(reset, result, atol=1e-7)

    info["steps"][:] = 1
    info["head_level_bias_ema"][:] = 0
    info["moving"][:] = True
    moving = MicroDuckEnlarged112WalkFlatEnv._head_level_bias_penalty(
        owner, SimpleNamespace(info=info)
    )
    np.testing.assert_allclose(moving, result * 2, atol=1e-7)


def test_course_changes_one_learning_dimension_and_preserves_physics():
    configs = []
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[4] / "conf/ppo"), version_base="1.3"
    ):
        for lesson in ("limits", "lateral", "balanced", "smooth", "idle", "switch"):
            cfg = compose(
                config_name="config_mlx",
                overrides=[f"task=microduck_enlarged_112_walk_flat/mujoco_{lesson}"],
            )
            configs.append(OmegaConf.to_container(cfg, resolve=True))
    limits, lateral, balanced, smooth, idle, switch = configs
    for cfg in configs:
        assert cfg["algo"]["num_envs"] == 256
        assert cfg["env"]["joint_limit_termination_margin_rad"] == 0.005
        assert cfg["env"]["joint_limit_barrier_scale"] == 25
        assert cfg["reward"]["action_rate_schedule_steps"] == [0]
        assert all(p > 0 for p in cfg["env"]["command_mode_probabilities"])
        assert sum(cfg["env"]["command_mode_probabilities"]) == pytest.approx(1)
        expected_algo = {**limits["algo"], "policy": dict(limits["algo"]["policy"])}
        if cfg is not limits:
            expected_algo["policy"].update(init_noise_std=0.15, resume_noise_std=0.15)
        assert cfg["algo"] == expected_algo
        expected_reward = {**limits["reward"], "scales": dict(limits["reward"]["scales"])}
        if cfg is not limits:
            expected_reward["scales"]["track_linear_velocity"] = 20.0
        assert cfg["reward"] == expected_reward
        assert {
            k: v
            for k, v in cfg["env"].items()
            if k
            not in (
                "command_mode_probabilities",
                "command_resample_interval_s",
                "linear_velocity_tracking_tau_s",
            )
        } == {
            k: v
            for k, v in limits["env"].items()
            if k
            not in (
                "command_mode_probabilities",
                "command_resample_interval_s",
                "linear_velocity_tracking_tau_s",
            )
        }
    assert {**limits["env"], "linear_velocity_tracking_tau_s": 0.25} == balanced["env"]
    assert balanced["env"] == smooth["env"]
    assert lateral["env"]["command_mode_probabilities"][3] == 0.6
    assert idle["env"]["command_mode_probabilities"][0] == 0.2
    assert idle["env"]["command_resample_interval_s"] == 8
    assert switch["env"]["command_resample_interval_s"] == 3


def test_feasibility_course_changes_only_existing_target_limit_signal():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[4] / "conf/ppo"), version_base="1.3"
    ):
        profiles = [
            OmegaConf.to_container(
                compose(config_name="config_mlx", overrides=[
                    f"task=microduck_enlarged_112_walk_flat/mujoco_{lesson}"
                ]), resolve=True
            )
            for lesson in ("limits", "feasible")
        ]
    baseline, candidate = profiles
    assert candidate["env"].pop("raw_target_feasibility_scale_rad") == 0.05
    assert candidate == baseline


def test_raw_target_penalty_covers_all_joints_before_protection():
    env = make_env("walk", 3, {"raw_target_feasibility_scale_rad": 0.05})
    try:
        raw = np.zeros((3, 14), np.float32)
        ctx = SimpleNamespace(info={"v112_reward_raw_action": raw})
        np.testing.assert_array_equal(env._raw_target_feasibility_penalty(ctx), 0)
        for joint in range(14):
            for bound, direction in ((env._v110_safe_joint_low, -1),
                                     (env._v110_safe_joint_high, 1)):
                raw[:] = 0
                raw[:, joint] = bound[joint] - env.default_angles[joint]
                raw[:, joint] += direction * np.array([0, 0.05, 0.20])
                np.testing.assert_allclose(
                    env._raw_target_feasibility_penalty(ctx), [0, 1, 16], rtol=3e-6, atol=1e-6
                )
        # The old penalty input may already have been governed: it must not
        # hide the raw request. This reward never edits actions or physics.
        state = env.init_state()
        before_q = env.get_dof_pos().copy()
        before_raw = raw.copy()
        ctx.info["motor_joint_target_limit_excess"] = np.zeros(3)
        assert env._raw_target_feasibility_penalty(ctx)[2] > 15.9
        np.testing.assert_array_equal(raw, before_raw)
        np.testing.assert_array_equal(env.get_dof_pos(), before_q)
        assert state.obs["obs"].shape == (3, 61)
    finally:
        env.close()


def test_phase_foot_height_reference_alternates_and_is_reward_only():
    env = make_env("walk", 4)
    try:
        state = env.init_state()
        quarter_steps = round(0.25 * env._phase_gait_period / env.cfg.ctrl_dt)
        three_quarter_steps = round(0.75 * env._phase_gait_period / env.cfg.ctrl_dt)
        steps = np.array([quarter_steps, three_quarter_steps, quarter_steps, quarter_steps])
        target = env._phase_foot_height_targets(steps)
        height = target.copy()
        height[2] = 0.0
        commands = np.full((4, 3), [0.2, 0.0, 0.0], np.float32)
        commands[3] = 0.0
        info = {"steps": steps.copy(), "foot_height": height, "commands": commands}
        before_qpos = env.get_dof_pos().copy()
        before = {key: value.copy() for key, value in info.items()}

        reward = env._phase_foot_height_tracking(SimpleNamespace(info=info))

        peak = env._reward_cfg.foot_height_target
        np.testing.assert_allclose(target[0], [peak, 0.0], atol=1e-7)
        np.testing.assert_allclose(target[1], [0.0, peak], atol=1e-7)
        np.testing.assert_allclose(reward[:2], 1.0, atol=1e-7)
        assert 0.0 < reward[2] < 0.2
        assert reward[3] == 0.0
        np.testing.assert_array_equal(env.get_dof_pos(), before_qpos)
        for key in info:
            np.testing.assert_array_equal(info[key], before[key])
        assert state.obs["obs"].shape == (4, 61)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [("phase_gait_period_s", 0.0), ("phase_foot_height_std_m", np.nan)],
)
def test_phase_foot_height_parameters_must_be_positive_and_finite(field, value):
    with pytest.raises(ValueError, match=field):
        make_env("walk", 1, {field: value})


def test_lateral_course_preserves_plant_and_prioritizes_translation():
    profiles = []
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[4] / "conf/ppo"), version_base="1.3"
    ):
        for lesson in ("limits", "lateral", "balanced"):
            cfg = compose(
                config_name="config_mlx",
                overrides=[f"task=microduck_enlarged_112_walk_flat/mujoco_{lesson}"],
            )
            profiles.append(OmegaConf.to_container(cfg, resolve=True))
    baseline, lateral, balanced = profiles
    assert balanced["env"] == {**baseline["env"], "linear_velocity_tracking_tau_s": 0.25}
    assert balanced["reward"] == lateral["reward"]
    assert lateral["reward"]["scales"]["track_linear_velocity"] == 20.0
    lateral["reward"]["scales"]["track_linear_velocity"] = baseline["reward"]["scales"][
        "track_linear_velocity"
    ]
    assert lateral["env"]["command_mode_probabilities"] == [0.05, 0.1, 0.1, 0.6, 0.1, 0.05]
    lateral["env"]["command_mode_probabilities"] = baseline["env"]["command_mode_probabilities"]
    assert lateral["env"].pop("linear_velocity_tracking_tau_s") == 0.25
    assert lateral["algo"]["policy"]["init_noise_std"] == 0.15
    assert lateral["algo"]["policy"]["resume_noise_std"] == 0.15
    lateral["algo"]["policy"].update(
        init_noise_std=baseline["algo"]["policy"]["init_noise_std"],
        resume_noise_std=baseline["algo"]["policy"]["resume_noise_std"],
    )
    assert lateral == baseline


def test_walk_command_switch_preserves_physics_and_updates_actor(monkeypatch):
    env = make_env("walk", 2, {"command_resample_interval_s": 0.04})
    try:
        state = env.init_state()
        command = np.array([[0.1, -0.1, 0.3], [-0.1, 0.1, -0.3]], np.float32)
        before = state.info["commands"].copy()
        monkeypatch.setattr(env._v112_walk_reset, "_sample_commands", lambda *_: command)

        def forbid_state_rewrite(*args, **kwargs):
            raise AssertionError("Command changes cannot rewrite physical state")

        monkeypatch.setattr(env._backend, "set_state", forbid_state_rewrite)
        state = env.step(np.zeros((2, 14), np.float32))
        np.testing.assert_array_equal(state.info["commands"], before)
        state = env.step(np.zeros((2, 14), np.float32))
        np.testing.assert_array_equal(state.info["commands"], command)
        np.testing.assert_array_equal(state.obs["obs"][:, 48:51], command)
        assert np.all(state.info["steps"] == 2)
    finally:
        env.close()


def test_evaluation_and_render_routes():
    from scripts.evaluate_microduck_enlarged_walk import PROFILES
    from scripts.render_microduck_enlarged_policy import TASKS

    assert PROFILES["enlarged_112"].task_name == "MicroDuckEnlarged112WalkFlat"
    assert PROFILES["enlarged_112_stand"].task_name == "MicroDuckEnlarged112StandFlat"
    assert TASKS["v112-walk"][0] == "microduck_enlarged_112_walk_flat/mujoco"
    assert TASKS["v112-stand"][0] == "microduck_enlarged_112_stand_flat/mujoco"


def test_asset_inertia_limits_and_actuator_order():
    model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "scene_flat.xml"))
    assert (model.nq, model.nv, model.nu, model.nbody) == (21, 20, 14, 22)
    assert model.body_mass.sum() == pytest.approx(8.8933)
    assert np.all(model.body_inertia[1:] > 0)
    assert np.all(model.body_inertia[1:].sum(axis=1) >= 2 * model.body_inertia[1:].max(axis=1))
    assert tuple(model.joint(int(j)).name for j in model.actuator_trnid[:, 0]) == JOINT_NAMES
    np.testing.assert_allclose(model.jnt_axis[1:, :2], 0)
    np.testing.assert_allclose(
        model.jnt_axis[1:, 2], [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1]
    )
    np.testing.assert_allclose(model.actuator_forcerange, [[-23.5, 23.5]] * 14)
    np.testing.assert_allclose(model.actuator_ctrlrange, [[-10, 10]] * 14)
    np.testing.assert_allclose(model.dof_armature[6:], 0.0288)
    np.testing.assert_allclose(model.dof_damping[6:], 0.1)
    np.testing.assert_allclose(model.dof_frictionloss[6:], 0.1)
    home = model.key("home").qpos[7:]
    assert np.all(home > model.jnt_range[1:, 0])
    assert np.all(home < model.jnt_range[1:, 1])
    np.testing.assert_allclose(model.key("home").ctrl, home)
    assert ET.parse(ASSET_ROOT / "microduck_enlarged_112.xml").find("keyframe") is None


def test_native_xml_is_only_a_plant_shell_not_the_fitted_motor():
    model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "scene_flat.xml"))
    old = mujoco.MjModel.from_xml_path(
        str(ASSET_ROOT.parent / "microduck_enlarged_111/scene_flat.xml")
    )
    for field in (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
    ):
        np.testing.assert_allclose(getattr(model, field), getattr(old, field))


@pytest.mark.parametrize("task", ["stand", "walk"])
def test_motor_configuration_changes_only_response_time_from_v111(task):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[4] / "conf/ppo"), version_base="1.3"
    ):
        old = compose(
            config_name="config_mlx", overrides=[f"task=microduck_enlarged_111_{task}_flat/mujoco"]
        )
        new = compose(
            config_name="config_mlx", overrides=[f"task=microduck_enlarged_112_{task}_flat/mujoco"]
        )
    for field in (
        "motor_constraints",
        "full_motor_dr",
        "domain_rand",
        "control_config",
    ):
        assert OmegaConf.to_container(old.env[field], resolve=True) == OmegaConf.to_container(
            new.env[field], resolve=True
        )
    expected_dynamics = OmegaConf.to_container(old.env.identified_dynamics, resolve=True)
    expected_dynamics["response_time_constant_range_s"] = [0.01, 0.01]
    assert OmegaConf.to_container(new.env.identified_dynamics, resolve=True) == expected_dynamics
    assert new.env.reset_base_qvel_limit == 0.0


@pytest.mark.parametrize("task", ["stand", "walk"])
def test_rollout_and_control_contract(task):
    np.random.seed(112)
    env = make_env(task)
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (4, 61)
        assert env.obs_groups_spec == {"obs": 61}
        assert env.action_space.shape == (14,)
        kp, kd = env._backend.get_actuator_gains()
        np.testing.assert_allclose(kp, 30)
        np.testing.assert_allclose(kd, 2)
        assert env._motor_constraints_enabled and env._identified_dynamics_enabled
        assert env.cfg.identified_dynamics.first_order_response
        np.testing.assert_allclose(env._identified_response_time_s, 0.01, rtol=0, atol=1e-9)
        assert all(spec is GF43X40_10 for spec in env._motor_specs)
        assert env.cfg.full_motor_dr.enabled
        np.testing.assert_allclose(env.get_dof_vel(), 0)
        np.testing.assert_allclose(env._backend.get_base_lin_vel(), 0)
        assert np.all(env.default_angles > env._v110_safe_joint_low)
        assert np.all(env.default_angles < env._v110_safe_joint_high)
        limits = env._backend.get_joint_range()
        nominal_indices = [0, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13]
        np.testing.assert_allclose(
            (env._v110_safe_joint_low - limits[:, 0])[nominal_indices],
            np.deg2rad(5),
            atol=2e-7,
        )
        assert env._v110_safe_joint_high[1] == pytest.approx(
            limits[1, 1] - np.deg2rad(env.cfg.hip_roll_outer_limit_margin_deg)
        )
        assert env._v110_safe_joint_low[10] == pytest.approx(
            limits[10, 0] + np.deg2rad(env.cfg.hip_roll_outer_limit_margin_deg)
        )
        # Driver HOME centers the neck; the full five-degree request inset
        # now fits, unlike the historical neck=-0.25 rad reference.
        assert env._v110_safe_joint_low[5] - limits[5, 0] == pytest.approx(np.deg2rad(5))
        # Velocity governor and fitted motor own their separate request/torque boundaries.
        for joint in range(14):
            action = np.zeros((4, 14), np.float32)
            action[:, joint] = 0.01
            env.apply_action(action, state)
            expected = np.broadcast_to(env.default_angles, (4, 14)).copy()
            expected[:, joint] += 0.01
            np.testing.assert_allclose(
                state.info["motor_requested_target_rad"], expected, atol=1e-7
            )
        # Driver HOME must support a continuous ten-second zero-action rollout
        # through the fitted motor and DR. Explicit reset remains checked below.
        saw_support = np.zeros((4, 2), dtype=bool)
        for _ in range(500):
            state = env.step(np.zeros((4, 14), np.float32))
            assert np.isfinite(state.obs["obs"]).all() and np.isfinite(state.reward).all()
            saw_support |= state.info["foot_contact"]
            torque = state.info["motor_fit_net_torque_nm"]
            assert np.all(torque >= state.info["motor_substep_torque_lower_nm"] - 1e-6)
            assert np.all(torque <= state.info["motor_substep_torque_upper_nm"] + 1e-6)
            assert not np.any(state.terminated)
            qpos = env.get_dof_pos()
            assert np.all(qpos > limits[:, 0]) and np.all(qpos < limits[:, 1])
        assert np.all(saw_support)
        obs, info = env.reset(np.array([0, 2]))
        assert isinstance(obs, dict) and isinstance(info, dict)
        np.testing.assert_allclose(env._identified_response_time_s, 0.01, rtol=0, atol=1e-9)
    finally:
        env.close()


def test_startup_state_is_integrated_from_rest_without_state_rewrites(monkeypatch):
    np.random.seed(112)
    env = make_env("stand", 1)
    try:
        env.init_state()
        samples = []
        original = env._motor_pre_step_control

        def capture(backend, ctrl):
            target = original(backend, ctrl)
            samples.append((env.get_dof_pos()[0].copy(), env.get_dof_vel()[0].copy()))
            return target

        def forbid_state_rewrite(*args, **kwargs):
            raise AssertionError("Policy startup must not teleport physical state")

        monkeypatch.setattr(env._backend, "set_state", forbid_state_rewrite)
        env._backend.set_pre_step_control(capture)
        for _ in range(50):
            env.step(np.full((1, 14), 0.005, np.float32))
        q = np.array([item[0] for item in samples])
        v = np.array([item[1] for item in samples])
        np.testing.assert_allclose(v[0], 0)
        np.testing.assert_allclose(np.diff(q, axis=0), env.cfg.sim_dt * v[1:], atol=2e-6)
    finally:
        env.close()


def test_walk_hip_yaw_request_has_deployable_rate_limit():
    env = make_env("walk", 1)
    try:
        state = env.init_state()
        action = np.zeros((1, 14), np.float32)
        action[:, [0, 9]] = [0.5, -0.5]
        previous = np.zeros(2, np.float32)
        startup_delta = env.cfg.hip_yaw_target_rate_limit_rad_s * env.cfg.ctrl_dt
        running_delta = env.cfg.hip_yaw_running_rate_limit_rad_s * env.cfg.ctrl_dt
        for _ in range(20):
            state = env.step(action)
            governed = state.info["v112_governed_actions"][0, [0, 9]]
            control_step = int(state.info["steps"][0]) - 1
            expected_delta = (
                startup_delta if control_step < env._hip_yaw_startup_steps else running_delta
            )
            np.testing.assert_array_less(np.abs(governed - previous), expected_delta + 1e-7)
            previous = governed.copy()
        np.testing.assert_allclose(previous, [0.5, -0.5], atol=2e-6)
        np.testing.assert_allclose(state.info["v112_policy_actions_raw"][0, [0, 9]], [0.5, -0.5])
    finally:
        env.close()


def test_walk_smoothing_prices_raw_actions_including_reset_first_frame():
    env = make_env("walk", 2)
    try:
        state = env.init_state()
        action = np.zeros((2, 14), np.float32)
        action[:, 0] = 0.5
        state = env.step(action)
        penalty = env._reward_fns["action_rate"](SimpleNamespace(info=state.info))
        np.testing.assert_allclose(penalty, 0.25)
        np.testing.assert_allclose(state.info["v112_governed_actions"][:, 0], 0.01)
        state = env.step(action)
        np.testing.assert_allclose(env._raw_action_rate(SimpleNamespace(info=state.info)), 0)
        # Simulate the reset-row history boundary with stale cached actions.
        state.info["steps"][0] = 0
        env.apply_action(action, state)
        np.testing.assert_allclose(
            env._raw_action_rate(SimpleNamespace(info=state.info)), [0.25, 0]
        )
        for step in (0, 12000, 36000):
            env._control_step = step
            env._update_action_rate_curriculum()
            assert env._reward_cfg.scales["action_rate"] == -1.0
    finally:
        env.close()


def test_joint_limit_clearance_penalty_tracks_physical_position():
    env = make_env("walk", 2)
    try:
        limits = env._backend.get_joint_range()
        home = np.broadcast_to(env.default_angles, (2, 14)).copy()
        qpos = home.copy()
        qpos[1, 10] = limits[10, 0] + 0.01
        penalty = env._joint_limit_clearance_penalty(SimpleNamespace(dof_pos=qpos))
        assert penalty[0] == pytest.approx(0.0)
        assert penalty[1] == pytest.approx(0.81)
    finally:
        env.close()


@pytest.mark.parametrize("task", ["stand", "walk"])
def test_clearance_reward_covers_every_joint_and_keeps_home_unpenalized(task):
    env = make_env(task, 2)
    try:
        limits = env._backend.get_joint_range()
        home = np.broadcast_to(env.default_angles, (2, 14)).copy()
        reserve = env._v112_joint_clearance_reserve
        assert np.all(reserve > 0)
        assert reserve[5] == pytest.approx(0.1, abs=1e-5)
        for joint in range(14):
            for side, sign in ((0, 1), (1, -1)):
                qpos = home.copy()
                qpos[1, joint] = limits[joint, side] + sign * reserve[joint] * 0.5
                cost = env._joint_limit_clearance_penalty(SimpleNamespace(dof_pos=qpos))
                np.testing.assert_allclose(cost, [0, 0.25], atol=1e-5)
                qpos[1, joint] = limits[joint, side] - sign * 0.01
                cost = env._joint_limit_clearance_penalty(SimpleNamespace(dof_pos=qpos))
                assert cost[0] == 0 and cost[1] > 1
    finally:
        env.close()


def test_clearance_barrier_prices_brief_neck_limit_excursions():
    env = make_env("walk", 4, {"joint_limit_barrier_scale": 25.0})
    try:
        limits = env._backend.get_joint_range()
        qpos = np.broadcast_to(env.default_angles, (4, 14)).copy()
        margins = np.array([0.015, 0.005, 0, -0.005])
        qpos[:, 5] = limits[5, 0] + margins
        cost = env._joint_limit_clearance_penalty(SimpleNamespace(dof_pos=qpos))
        expected = np.square(1 - margins / env._v112_joint_clearance_reserve[5])
        expected += 25 * np.square(np.maximum(1 - margins / 0.01, 0))
        np.testing.assert_allclose(cost, expected, atol=1e-4)
        assert np.all(np.diff(cost) > 0)
        assert cost[1] * -5 < -31
        assert cost[2] * -5 <= -130
    finally:
        env.close()


def test_clearance_termination_marks_failure_without_repairing_state(monkeypatch):
    env = make_env("walk", 2, {"joint_limit_termination_margin_rad": 0.005})
    try:
        env.init_state()
        state = env.step(np.zeros((2, 14), np.float32))
        actual = env._backend.get_dof_pos().copy()
        qpos = np.broadcast_to(env.default_angles, (2, 14)).copy()
        qpos[:, 5] = env._backend.get_joint_range()[5, 0] + [0.003, 0.007]
        monkeypatch.setattr(env, "get_dof_pos", lambda: qpos)
        state = env.update_state(state)
        np.testing.assert_array_equal(state.info["v112_joint_limit_termination"], [True, False])
        np.testing.assert_array_equal(state.terminated, [True, False])
        np.testing.assert_array_equal(env._backend.get_dof_pos(), actual)
    finally:
        env.close()


def test_mean_velocity_reward_resets_history_without_filtering_physics(monkeypatch):
    env = make_env("walk", 1, {"linear_velocity_tracking_tau_s": 0.25})
    try:
        env.init_state()
        state = env.step(np.zeros((1, 14), np.float32))
        physical_velocity = env._backend.get_base_lin_vel().copy()
        measured = np.array([[0, -0.12, 0]], np.float32)
        monkeypatch.setattr(env, "get_local_linvel", lambda: measured)
        state.info["steps"][:] = 1
        state.info["commands"][:] = [0, 0.12, 0]
        state.info["v112_tracking_linvel"][:] = [0, 0.12, 0]
        state = env.update_state(state)
        expected = 0.12 + env._linear_tracking_alpha * (-0.24)
        np.testing.assert_allclose(state.info["v112_tracking_linvel"], [[0, expected, 0]])
        reward = env._reward_fns["track_linear_velocity"](SimpleNamespace(info=state.info))
        np.testing.assert_allclose(reward, np.exp(-((0.12 - expected) ** 2) / 0.02))
        state.info["steps"][:] = 0
        state.info["v112_tracking_linvel"][:] = 99
        state = env.update_state(state)
        np.testing.assert_allclose(state.info["v112_tracking_linvel"], measured)
        np.testing.assert_array_equal(env._backend.get_base_lin_vel(), physical_velocity)
        assert state.obs["obs"].shape == (1, 61)
    finally:
        env.close()


def test_lateral_progress_requires_correct_signed_motion():
    env = make_env("walk", 1)
    try:
        command = np.array([[0, 0.12, 0], [0, -0.12, 0], [0, -0.12, 0],
                            [0, 0.12, 0], [0.3, 0, 0], [0, 0, 0],
                            [0, 0.12, 0]], dtype=np.float32)
        velocity = np.zeros((7, 3), dtype=np.float32)
        velocity[:, 1] = [0.12, -0.12, 0.12, 0, 0.12, 0.12, 0.3]
        ctx = SimpleNamespace(info={"commands": command}, linvel=velocity)
        np.testing.assert_allclose(env._lateral_command_progress(ctx), [1, 1, -1, 0, 0, 0, 1])
        # Instantaneous sway must not override the reward-only averaged velocity.
        env._linear_tracking_alpha = 0.1
        ctx.info["v112_tracking_linvel"] = -velocity
        np.testing.assert_allclose(env._lateral_command_progress(ctx), [-1, -1, 1, 0, 0, 0, -1])
    finally:
        env.close()


def test_rejected_walk_action_does_not_poison_raw_reward_history():
    env = make_env("walk", 1)
    try:
        action = np.zeros((1, 14), np.float32)
        env.init_state()
        state = env.step(action)
        history = state.info["v112_reward_raw_action"].copy()
        for bad in (np.full((1, 14), np.nan), np.zeros((1, 13))):
            with pytest.raises(ValueError, match="finite with shape"):
                env.apply_action(bad, state)
            np.testing.assert_array_equal(state.info["v112_reward_raw_action"], history)
        state = env.step(action)
        assert np.isfinite(state.reward).all()
        np.testing.assert_array_equal(env._raw_action_rate(SimpleNamespace(info=state.info)), 0)
    finally:
        env.close()


def test_imu_frame_in_live_and_reset_observations(monkeypatch):
    env = make_env("stand", 1)
    try:
        m = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "scene_flat.xml"))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        d.qvel[3:6] = [0.4, -0.5, 0.6]
        mujoco.mj_forward(m, d)
        gyro = d.sensor("imu_ang_vel").data[None].copy()
        monkeypatch.setattr(env._backend, "get_sensor_data", lambda _: gyro)
        np.testing.assert_allclose(env.get_gyro()[0], d.qvel[3:6], atol=1e-6)
        captured = {}
        monkeypatch.setattr(
            env, "_compute_obs", lambda info, gyro, *args: captured.update(gyro=gyro)
        )
        env._make_domain_randomization_provider()._compute_reset_obs(
            env,
            np.array([0]),
            {},
            np.zeros((1, 3)),
            gyro,
            np.zeros((1, 3)),
            np.zeros((1, 14)),
            np.zeros((1, 14)),
        )
        np.testing.assert_allclose(captured["gyro"][0], d.qvel[3:6], atol=1e-6)
    finally:
        env.close()
