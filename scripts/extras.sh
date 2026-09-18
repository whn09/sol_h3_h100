#!/usr/bin/env bash
# The two arms matrix.sh deliberately leaves out, because neither answers "how does one H100 compare
# to upstream's one-B300 row" -- they answer the two questions that come after it.
#
#   bash extras.sh                     # both, ~12 min
#   ARMS="adapter" bash extras.sh
#
#   adapter  -> the real FastH3 four-step LoRA, merged. Two things at once: the first MP4 from this
#               box that is a QUALITY artifact rather than a latency artifact (base weights at 5
#               scheduler points are the right speed and the wrong video), and a check that
#               `--lora-mode merged` is latency-neutral as claimed -- it folds into the weights
#               before the loop, so d5_adapter should land on top of d5_step5's 29.3 s. If it does
#               not, every no-adapter number in RESULTS.md is suspect.
#   p480     -> 864x480. 1344x768 is a module constant in Sol-H3's engine but a per-request argument
#               to the pipeline underneath, so the smaller canvas costs nothing to reach. It is
#               2.49x fewer pixels, and the point is whether the H100 gap closes there: if the 768p
#               number is dominated by attention over a sequence one card cannot shard (no Ulysses
#               at world_size=1), 480p should scale better than linearly in pixels. It is also the
#               canvas the 8-card work used, so it is the one number comparable to that repo.
#               15 s at 480p is included because it is the longest clip this card can serve at all.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/sol}
PY=${PY:-$ROOT/venv/bin/python}
BENCH=$ROOT/scripts/bench_h100.py
REPEATS=${REPEATS:-2}
ARMS=${ARMS:-"adapter p480"}
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/vdn/hf}
ADAPTER=${ADAPTER:-$(echo "$HF_HOME"/hub/models--FastVideo--FastVideo-FastH3-4-step-Preview-v1-LoRA/snapshots/*/dense-datafree/adapter_model.safetensors)}
export PYTHONPATH=$ROOT/Sol-H3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {  # run <tag> <duration> <steps> [extra args...]
  local tag=$1 duration=$2 steps=$3; shift 3
  echo "=== $tag: ${duration}s, $steps points, $* $(date -u +%T) ==="
  $PY "$BENCH" --duration "$duration" --steps "$steps" --repeats "$REPEATS" --warmup 1 "$@" \
    --json "$ROOT/logs/$tag.json" --output "$ROOT/out/$tag.mp4" \
    > "$ROOT/logs/$tag.log" 2>&1
  grep -E "^  (warmup|request)|^median" "$ROOT/logs/$tag.log" | tail -5
}

for arm in $ARMS; do
  case $arm in
    adapter)
      [ -f "$ADAPTER" ] || { echo "no adapter at $ADAPTER"; continue; }
      echo "adapter: $(du -hL "$ADAPTER" | cut -f1)  $ADAPTER"
      run d5_adapter      5 5 --adapter "$ADAPTER" --lora-mode merged ;;
    p480)
      run d5_480p         5 5 --width 864 --height 480
      run d15_480p       15 5 --width 864 --height 480 ;;
    *) echo "unknown arm $arm" ;;
  esac
done
echo "=== extras done $(date -u +%T) ==="
