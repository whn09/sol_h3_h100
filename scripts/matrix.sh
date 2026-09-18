#!/usr/bin/env bash
# The single-H100 arms of upstream's own table, in the order that answers the most per GPU-hour.
#
#   bash matrix.sh                 # everything below, sequentially, ~35 min
#   ARMS="d10 d15" bash matrix.sh  # a subset
#
# WHY THESE ARMS. Upstream publishes three one-GPU rows per length -- Sol-H3 four-forward, SGLang
# 49-forward, Diffusers 49-forward -- all on B300. Two questions follow for an H100, and each arm
# answers exactly one:
#
#   d5 / d10 / d15 at 5 scheduler points   -> the card gap. Same profile, same canvas, same seed,
#                                             same four forwards as upstream's 13.745 / 37.813 /
#                                             52.260 s. The only variable left is B300 vs H100.
#   base50 at 5 s                          -> the schedule gap, measured on THIS code path rather
#                                             than against someone else's runtime. 49 forwards of
#                                             the same fused kernels on the same card is what
#                                             separates "the adapter is shorter" from "the engine
#                                             is faster"; comparing 4-forward Sol-H3 to SGLang
#                                             conflates the two.
#
# base50 is last because it is the expensive one: 49 forwards at ~5.2 s each is ~4.5 min per
# request, so one warmup and two requests is ~14 min of the ~35.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/sol}
PY=${PY:-$ROOT/venv/bin/python}
BENCH=$ROOT/scripts/bench_h100.py
ADAPTER=${ADAPTER:-}                     # FastH3 four-step LoRA, when a token exists for it
REPEATS=${REPEATS:-2}
ARMS=${ARMS:-"d5 d10 d15 base50"}

export HF_HOME=${HF_HOME:-/opt/dlami/nvme/vdn/hf}
export PYTHONPATH=$ROOT/Sol-H3
# Online quantisation and a 60 GiB conditioner in the same process fragment the arena; the 8-card
# work needed this flag for the same reason and it costs nothing when it is not needed.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {  # run <tag> <duration> <steps>
  local tag=$1 duration=$2 steps=$3
  local extra=()
  [ -n "$ADAPTER" ] && extra=(--adapter "$ADAPTER")
  echo "=== $tag: ${duration}s, $steps scheduler points ($((steps - 1)) forwards) $(date -u +%T) ==="
  $PY "$BENCH" --duration "$duration" --steps "$steps" --repeats "$REPEATS" --warmup 1 \
    "${extra[@]}" \
    --json "$ROOT/logs/$tag.json" --output "$ROOT/out/$tag.mp4" \
    > "$ROOT/logs/$tag.log" 2>&1
  grep -E "^  (warmup|request)|^median|peak reserved" "$ROOT/logs/$tag.log" | tail -6
}

for arm in $ARMS; do
  case $arm in
    d5)     run d5_step5   5  5 ;;
    d10)    run d10_step5 10  5 ;;
    d15)    run d15_step5 15  5 ;;
    base50) run d5_step50  5 50 ;;
    *)      echo "unknown arm $arm" ;;
  esac
done
echo "=== matrix done $(date -u +%T) ==="
