#!/usr/bin/env bash
# Full zero-shot benchmark: baselines, prompt ablation, chunk-length ablation.
# Usage: scripts/run_benchmark.sh [episodes]   (run from the repo root, ManiSkill env active)
set -euo pipefail
EPISODES="${1:-20}"
VLA_PY="${VLA_PY:-$HOME/miniforge3/envs/smolvla/bin}"
PLAIN="pick up the red cube"
LIBERO="pick up the red block and place it on the plate"

pms-sim-server --render-mode rgb_array --lockstep --max-episode-steps 200 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 8

run() { "$VLA_PY/pms-vla" --episodes "$EPISODES" "$@" 2>&1 | grep -E "^\[|success|Saved"; }

run --policy zero
run --policy random
run --policy smolvla --task "$PLAIN" --name smolvla_plain_prompt_k10
run --policy smolvla --task "$LIBERO" --name smolvla_libero_prompt_k10 --videos 2
for k in 5 25 50; do
  run --policy smolvla --task "$LIBERO" --actions-per-chunk "$k" --name "smolvla_libero_prompt_k$k"
done

"$VLA_PY/pms-vla-report" results/zero_k10.json results/random_k10.json results/smolvla_*.json \
  | tee results/summary.md
