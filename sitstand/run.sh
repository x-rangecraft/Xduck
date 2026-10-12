#!/usr/bin/env bash
set -euo pipefail
archive_root=$(cd "$(dirname "$0")" && pwd)
cd "$archive_root/unilab"
export PYTHONPATH="$archive_root/runtime_snapshot:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
runtime_python=${XDUCK_PYTHON:-python3.10}
mode=${1:-verify}
mkdir -p ../outputs
case "$mode" in
  verify) uv run --no-project --python "$runtime_python" python ../verify_archive.py ;;
  train) uv run --no-project --python "$runtime_python" python scripts/train_rsl_rl.py --config-path "$archive_root/configs" --config-name resume_2000_to2500 ;;
  eval)
    for head_mode in curriculum fixed_zero; do
      uv run --no-project --python "$runtime_python" python scripts/evaluate_xduck_training_search.py --project . --checkpoint ../checkpoints/model_2499.pt --head-mode "$head_mode" --output "../outputs/eval_2499_$head_mode.json"
      uv run --no-project --python "$runtime_python" python scripts/reevaluate_xduck_geometry.py --model ../physics/pure_effective_newstand_linux.mjb --trajectory "../outputs/eval_2499_$head_mode.npz" --output "../outputs/eval_2499_${head_mode}_geometry.json" --stand-height .3198 --sit-height .2182
    done ;;
  *) echo 'Usage: run.sh [verify|train|eval]' >&2; exit 2 ;;
esac
