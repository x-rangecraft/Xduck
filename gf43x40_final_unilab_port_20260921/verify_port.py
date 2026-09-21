"""Verify this portable bundle without importing any installed UniLab."""
import hashlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "MANIFEST.json").read_text())
for name, expected in manifest["files_sha256"].items():
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"File checksum mismatch: {name}")

# Explicit package roots prevent an installed UniLab from masking missing files.
for name, relative in (("unilab", "src/unilab"),
                       ("unilab.actuators", "src/unilab/actuators")):
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / relative)]
    sys.modules[name] = module

from unilab.actuators.gf43x40 import GF43X40Parameters
import unilab.actuators.gf43x40 as imported

assert Path(imported.__file__).resolve().is_relative_to(ROOT)
parameters = GF43X40Parameters.from_bundle()
assert parameters.source_sha256 == manifest["source_parameters_sha256"]
assert parameters.observation["model"] == "mechanical_torque_proxy"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest

print(f"Verified {len(manifest['files_sha256'])} files; loading {imported.__file__}")
raise SystemExit(pytest.main([
    str(ROOT / "tests/actuators"), "-q", "-c", os.devnull,
    "--confcutdir", str(ROOT), "-p", "no:cacheprovider",
]))
