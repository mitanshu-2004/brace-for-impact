#!/usr/bin/env bash
# Render the full clip set: the force ladder for all three conditions, both training
# progressions, the projectile showcase, and two hero shots.
#
# One process per clip. Each mjlab environment build costs about a minute, so sweeping
# force inside a single process would be faster, but a crash then loses the whole batch
# instead of one file. Finished clips are skipped on a rerun.
#
# About an hour end to end on a T4.
#
# Usage:  bash scripts/render_all.sh [output_dir]
set -u

cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=${MUJOCO_GL:-egl} WANDB_MODE=disabled

OUT=${1:-clips}
WARNED=${WARNED_CKPT:-logs/rsl_rl/brace_warned/model_3999.pt}
UNWARNED=${UNWARNED_CKPT:-logs/rsl_rl/brace_unwarned/model_3999.pt}
mkdir -p "$OUT"

HD="--width 1920 --height 1080"
UHD="--width 2560 --height 1440"

run() {   # run <outfile> <script> <args...>
  local out="$1"; shift
  if [ -s "$out" ]; then echo "skip $out"; return; fi
  echo "== $out"
  if ! python "$@" --out "$out"; then
    echo "!! failed $out"
    # A partial file would be skipped as finished on the next run.
    rm -f "$out" "${out%.mp4}_short.mp4"
  fi
}

# Force ladder, same forces as eval.py so each clip has a measured fall rate beside it.
# 900 frames is 18 s, which carries 3 to 6 impulses at the 3-6 s resample schedule.
for F in 35 60 90 120 160 200 250; do
  run "$OUT/warned_${F}N.mp4" scripts/render.py \
      --robot warned --checkpoint "$WARNED" --force $F --frames 900 $HD
  run "$OUT/unwarned_${F}N.mp4" scripts/render.py \
      --robot unwarned --checkpoint "$UNWARNED" --force $F --frames 900 $HD
  run "$OUT/warning_removed_${F}N.mp4" scripts/render.py \
      --robot warned --checkpoint "$WARNED" --force $F --frames 900 --no-warning $HD
done

# All 81 checkpoints in order, 100 frames each: the policy learning to brace.
run "$OUT/warned_learning.mp4" scripts/render.py \
    --robot warned --force 250 --segment-frames 100 $HD \
    --checkpoint "$(dirname "$WARNED")"/model_*.pt
run "$OUT/unwarned_learning.mp4" scripts/render.py \
    --robot unwarned --force 250 --segment-frames 100 $HD \
    --checkpoint "$(dirname "$UNWARNED")"/model_*.pt

# A real ball instead of a commanded wrench. Also writes a _highlights cut.
run "$OUT/warned_ball.mp4" scripts/showcase.py \
    --robot warned --checkpoint "$WARNED" --frames 900 $UHD
run "$OUT/unwarned_ball.mp4" scripts/showcase.py \
    --robot unwarned --checkpoint "$UNWARNED" --frames 900 $UHD

run "$OUT/warned_250N_orbit.mp4" scripts/render.py \
    --robot warned --checkpoint "$WARNED" --force 250 --frames 900 \
    --orbit 12 --distance 1.8 --elevation -10 $UHD
run "$OUT/warned_250N_closeup.mp4" scripts/render.py \
    --robot warned --checkpoint "$WARNED" --force 250 --frames 600 \
    --distance 1.5 --elevation -3 --azimuth 200 $UHD

echo "done"
ls -la "$OUT"
