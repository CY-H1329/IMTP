#!/usr/bin/env bash
# Common env for selective_news on H100 (4 GPUs: 0,1,2,3)
set -euo pipefail
SN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SELECTIVE_BENCH_ROOT="${SELECTIVE_BENCH_ROOT:-$SN_ROOT}"
export PYTHONPATH="${SN_ROOT}/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$SN_ROOT/.cache/huggingface}"
mkdir -p "$SN_ROOT/results" "$SN_ROOT/logs" "$HF_HOME"

# Data: prefer local crops, else HF pack
if [[ -z "${NEWS_DATA:-}" ]]; then
  if [[ -d "$SN_ROOT/data/hf_news_pack" ]]; then
    export NEWS_DATA="$SN_ROOT/data/hf_news_pack"
  elif [[ -d "$SN_ROOT/data/crops" ]]; then
    export NEWS_DATA="$SN_ROOT/data"
  fi
fi
echo "[env] SELECTIVE_BENCH_ROOT=$SELECTIVE_BENCH_ROOT"
echo "[env] NEWS_DATA=${NEWS_DATA:-unset}"

# Large model weights on /dataset (root FS tight)
export HF_HOME="${HF_HOME:-/dataset/chanyeong/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
