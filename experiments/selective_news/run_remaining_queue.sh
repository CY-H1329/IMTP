#!/usr/bin/env bash
# Chain remaining tables after current InternVL 8-shard job finishes.
set -euo pipefail
SN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SN_ROOT"
source .venv/bin/activate
source env.sh
export NEWS_DATA="${NEWS_DATA:-/home/gpuuser/chanyeong/data/hf_news_pack}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPUS="${GPUS:-0,1,2,3}"

wait_internvl() {
  echo "[queue] waiting for internvl35 workers to finish..."
  while pgrep -f "run_news_tables.py --models internvl35" >/dev/null 2>&1; do
    sleep 60
  done
  # also wait parent run_4gpu if any
  while pgrep -f "run_4gpu.sh tables" >/dev/null 2>&1; do
    # only if still internvl-related parent
    sleep 30
    # break if no internvl children
    pgrep -f "run_news_tables.py --models internvl35" >/dev/null 2>&1 || break
  done
  echo "[queue] internvl done at $(date -Is)"
  ./run_4gpu.sh score || true
}

run_tables() {
  local wpg="$1"; shift
  local models="$1"; shift
  echo "[queue] WORKERS_PER_GPU=$wpg MODELS=$models"
  WORKERS_PER_GPU="$wpg" MODELS="$models" ./run_4gpu.sh tables
}

wait_internvl

# 8B-class: 2 workers / GPU (same as InternVL)
run_tables 2 "gemma12 gemma12 gemma12 gemma12"

# 27B: 1 worker / GPU
run_tables 1 "qwen38 qwen38 qwen38 qwen38"
run_tables 1 "gemma27 gemma27 gemma27 gemma27"

# method / strict / lang
N_ART_METHOD=50 MODELS="qwen3vl internvl35 qwen38 gemma12" ./run_4gpu.sh method
N_STRICT=25 MODELS="qwen3vl internvl35 qwen38 gemma12" ./run_4gpu.sh strict
MODELS="qwen3vl" ./run_4gpu.sh lang
./run_4gpu.sh score

echo "[queue] ALL DONE $(date -Is)"
