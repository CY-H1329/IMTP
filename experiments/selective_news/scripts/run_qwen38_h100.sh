#!/usr/bin/env bash
# Table 1 + Table 3 for Qwen3.8-27B on one (or more) H100s.
# Run from the unpacked pack root:  bash run_qwen38_h100.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export SELECTIVE_BENCH_ROOT="${SELECTIVE_BENCH_ROOT:-$ROOT/selective_bench}"
export PYTHONPATH="${ROOT}:${SELECTIVE_BENCH_ROOT}/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-qwen38}"
GOLD="${GOLD:-$ROOT/gold/news_eval.json}"
DEST="${DEST:-$ROOT/results/news_tables}"
SHARD="${SHARD:-0/1}"
LIMIT="${LIMIT:-0}"
GPU="${GPU:-0}"

mkdir -p "$DEST" logs
export CUDA_VISIBLE_DEVICES="$GPU"

extra=()
if [[ "$LIMIT" != "0" ]]; then
  extra+=(--limit "$LIMIT")
fi

echo "Qwen3.8-27B  gpu=$GPU shard=$SHARD gold=$GOLD dest=$DEST" | tee -a logs/qwen38.log
python selective_bench/scripts/run_news_tables.py \
  --models "$MODEL" \
  --gold "$GOLD" \
  --dest-dir "$DEST" \
  --shard "$SHARD" \
  "${extra[@]}" \
  2>&1 | tee -a logs/qwen38.log
