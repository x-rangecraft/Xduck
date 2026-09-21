from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType

import pytest

import mujoco_uni.mujoco_runtime.version_control as version_control
from mujoco_uni.mujoco_runtime.version_control import (
    MUJOCO_UNI_VERSION_ENV,
    MuJoCoEnv,
    discover_mujoco_envs,
    requested_mujoco_version,
    run_in_env,
)


def test_requested_mujoco_version_reads_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MUJOCO_UNI_VERSION_ENV, raising=False)
    assert requested_mujoco_version() is None

    monkeypatch.setenv(MUJOCO_UNI_VERSION_ENV, " 3.10 ")
    assert requested_mujoco_version() == "3.10"

    monkeypatch.setenv(MUJOCO_UNI_VERSION_ENV, "")
    assert requested_mujoco_version() is None


def test_verify_mujoco_runtime_matches_minor_or_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MjModel:
        _address = 1

        @staticmethod
        def _from_model_ptr(_ptr):
            return None

    module = ModuleType("mujoco")
    module.__version__ = "3.8.1"
    module.MjModel = _MjModel

    monkeypatch.setenv(MUJOCO_UNI_VERSION_ENV, "3.8")
    assert version_control.verify_mujoco_runtime(module) == "3.8.1"

    monkeypatch.setenv(MUJOCO_UNI_VERSION_ENV, "3.8.1")
    assert version_control.verify_mujoco_runtime(module) == "3.8.1"

    monkeypatch.setenv(MUJOCO_UNI_VERSION_ENV, "3.8.0")
    with pytest.raises(ImportError, match="requested"):
        version_control.verify_mujoco_runtime(module)


def test_discover_mujoco_envs_keeps_venv_python_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlinks are not available")

    env_dir = tmp_path / ".venv-mj310"
    venv_python = version_control._python_path(env_dir)
    venv_python.parent.mkdir(parents=True)
    base_python = tmp_path / "base-python"
    base_python.write_text("")
    try:
        venv_python.symlink_to(base_python)
    except OSError:
        pytest.skip("symlinks are not supported on this filesystem")

    probed: list[Path] = []

    def fake_mujoco_version_from_python(python: Path) -> str:
        probed.append(python)
        return "3.10.0"

    monkeypatch.setattr(
        version_control,
        "_mujoco_version_from_python",
        fake_mujoco_version_from_python,
    )

    envs = discover_mujoco_envs([tmp_path], include_current=False)

    assert envs == [
        MuJoCoEnv(
            version="3.10.0",
            env_dir=env_dir,
            python=venv_python.absolute(),
        )
    ]
    assert probed == [venv_python.absolute()]


def test_run_in_env_uses_existing_explicit_env_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "custom-env"
    python = version_control._python_path(env_dir)
    python.parent.mkdir(parents=True)
    python.write_text("")

    monkeypatch.setattr(version_control, "_mujoco_version_from_python", lambda _python: "3.10.0")
    monkeypatch.setattr(version_control, "verify_env", lambda _env: None)
    recorded: list[tuple[list[str], dict[str, str]]] = []

    def fake_subprocess_run(
        command: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del check
        recorded.append((command, env))
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(version_control.subprocess, "run", fake_subprocess_run)

    result = run_in_env(["python", "train.py"], version="3.10", env_dir=env_dir)

    assert result == 7
    assert recorded[0][0] == [str(python), "train.py"]
    assert recorded[0][1][MUJOCO_UNI_VERSION_ENV] == "3.10.0"


def test_run_in_env_falls_back_to_existing_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env38 = _make_env(tmp_path, ".venv-mj38")
    _make_env(tmp_path, ".venv-mj310")

    def fake_mujoco_version_from_python(python: Path) -> str:
        return "3.8.1" if ".venv-mj38" in str(python) else "3.10.0"

    monkeypatch.setattr(version_control, "_mujoco_version_from_python", fake_mujoco_version_from_python)
    monkeypatch.setattr(version_control, "verify_env", lambda _env: None)
    recorded: list[tuple[list[str], dict[str, str]]] = []

    def fake_subprocess_run(
        command: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del check
        recorded.append((command, env))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(version_control.subprocess, "run", fake_subprocess_run)

    result = run_in_env(["python", "train.py"], version="3.9", env_root=tmp_path)

    assert result == 0
    assert recorded[0][0] == [str(version_control._python_path(env38)), "train.py"]
    assert recorded[0][1][MUJOCO_UNI_VERSION_ENV] == "3.8.1"
    output = capsys.readouterr().out
    assert "requested mujoco '3.9' was not found" in output
    assert "using mujoco 3.8.1" in output


def test_run_in_env_skips_unusable_requested_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env38 = _make_env(tmp_path, ".venv-mj38")
    env310 = _make_env(tmp_path, ".venv-mj310")

    def fake_mujoco_version_from_python(python: Path) -> str:
        return "3.8.1" if ".venv-mj38" in str(python) else "3.10.0"

    def fake_verify_env(env: MuJoCoEnv) -> None:
        if env.env_dir == env310:
            raise subprocess.CalledProcessError(1, [str(env.python), "-c", "probe"])

    monkeypatch.setattr(version_control, "_mujoco_version_from_python", fake_mujoco_version_from_python)
    monkeypatch.setattr(version_control, "verify_env", fake_verify_env)
    recorded: list[list[str]] = []

    def fake_subprocess_run(
        command: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del check, env
        recorded.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(version_control.subprocess, "run", fake_subprocess_run)

    assert run_in_env(["python", "train.py"], version="3.10", env_root=tmp_path) == 0
    assert recorded == [[str(version_control._python_path(env38)), "train.py"]]
    assert "requested mujoco '3.10' was found but unusable" in capsys.readouterr().out


def test_run_in_env_fails_when_no_env_exists(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No existing MuJoCo environments"):
        run_in_env(["python", "train.py"], version="3.10", env_dir=tmp_path / "missing")


def _make_env(root: Path, name: str) -> Path:
    env_dir = root / name
    python = version_control._python_path(env_dir)
    python.parent.mkdir(parents=True)
    python.write_text("")
    return env_dir
