#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/lemon/Issac/IsaacLab}"

export PYTHONPATH="${ROOT_DIR}/source/robot_lab:${PYTHONPATH:-}"
export GO2_X5_LOW_LEVEL_POLICY_PATH="${GO2_X5_LOW_LEVEL_POLICY_PATH:-${ROOT_DIR}/logs/rsl_rl/go2_x5_flat/2026-02-06_00-39-51/exported/policy.pt}"

exec "${ISAACLAB_ROOT}/isaaclab.sh" \
  -p "${ROOT_DIR}/scripts/tools/view_ground_pick.py" \
  --device cuda:0 \
  --num_envs 1 \
  --enable_cameras \
  "$@"
