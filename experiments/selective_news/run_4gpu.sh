#!/usr/bin/env bash
# Launch selective_news on GPUs 0,1,2,3
# Usage (from experiments/selective_news):
#   source env.sh && ./run_4gpu.sh smoke|e1|tables|method|lang|strict|score|all
set -euo pipefail
SN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SN_ROOT/env.sh"
cd "$SN_ROOT"
chmod +x scripts/*.sh 2>/dev/null || true

GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
GOLD="${GOLD:-$SN_ROOT/gold/news_eval.json}"
OUT="${OUT:-$SN_ROOT/results}"
LOG="$SN_ROOT/logs"
mkdir -p "$OUT" "$LOG" "$OUT/news_tables_act" "$OUT/bind_method" "$OUT/lang" "$OUT/strict_small" "$OUT/rule_yesno"
MODE="${1:-smoke}"
LIMIT="${LIMIT:-0}"   # 0 = all articles
# Pack N inference workers per physical GPU (tables). 2 fits ~18GB×2 on H100 80GB for 8B VLMs.
# Large models (qwen38 / gemma27): export WORKERS_PER_GPU=1
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

need_data() {
  if [[ -z "${NEWS_DATA:-}" ]]; then
    echo "ERROR: set NEWS_DATA (hf_news_pack root with crops/)" >&2
    echo "  export NEWS_DATA=\$HOME/data/hf_news_pack" >&2
    exit 1
  fi
  if [[ ! -d "$NEWS_DATA/crops" ]]; then
    echo "ERROR: NEWS_DATA/crops missing: $NEWS_DATA" >&2
    exit 1
  fi
}

run_bg() {
  local gpu="$1"; shift
  local name="$1"; shift
  echo "[launch] GPU=$gpu name=$name → $LOG/${name}.log"
  CUDA_VISIBLE_DEVICES="$gpu" setsid python "$@" \
    >"$LOG/${name}.log" 2>&1 &
  echo $! >"$LOG/${name}.pid"
}

wait_all() {
  echo "[wait] $(date -Is) pids=$(cat "$LOG"/*.pid 2>/dev/null | tr '\n' ' ')"
  wait || true
  echo "[wait done] $(date -Is)"
}

case "$MODE" in
  smoke)
    need_data
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python scripts/run_news_tables.py \
      --models qwen3vl --gold "$GOLD" --dest-dir "$OUT/smoke" --limit 4 \
      --shard 0/1 2>&1 | tee "$LOG/smoke.log"
    python scripts/score_news_tables.py --dest-dir "$OUT/smoke"
    ;;

  e1)
    # E1: verbal KNOW-rules (blank image). One model per GPU.
    MODELS="${MODELS:-qwen3vl internvl35 gemma12}"
    read -r -a M_ARR <<< "$MODELS"
    i=0
    for m in "${M_ARR[@]}"; do
      gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
      run_bg "$gpu" "e1_${m}" scripts/run_rule_yesno.py \
        --models "$m" --out "$OUT/rule_yesno/${m}.json"
      i=$((i+1))
    done
    wait_all
    python scripts/score_all_tables.py --results-dir "$OUT" --which e1
    ;;

  tables)
    # E2–E8: Decision probe + ACT (preserve / translate@gt); guided = execution
    # WORKERS_PER_GPU (default 2): launch N shards per physical GPU → better H100 util for 8B.
    need_data
    MODELS="${MODELS:-qwen3vl qwen3vl qwen3vl qwen3vl}"
    read -r -a M_ARR <<< "$MODELS"
    n_gpu="${#GPU_ARR[@]}"
    wpg=$((WORKERS_PER_GPU + 0))
    if [[ "$wpg" -lt 1 ]]; then wpg=1; fi
    n=$((n_gpu * wpg))
    DEST_T="${OUT}/news_tables_act"
    mkdir -p "$DEST_T"
    echo "[tables] GPUs=${GPUS} workers_per_gpu=${wpg} shards=${n} models=${MODELS}"
    for ((i=0; i<n; i++)); do
      gpu="${GPU_ARR[$((i % n_gpu))]}"
      m="${M_ARR[$((i % ${#M_ARR[@]}))]}"
      extra=()
      [[ "$LIMIT" != "0" ]] && extra+=(--limit "$LIMIT")
      run_bg "$gpu" "tables_${m}_s${i}of${n}" scripts/run_news_tables.py \
        --models "$m" --gold "$GOLD" --dest-dir "$DEST_T" \
        --shard "${i}/${n}" "${extra[@]}"
    done
    wait_all
    python scripts/score_news_tables.py --dest-dir "$DEST_T"
    python scripts/score_all_tables.py --results-dir "$OUT" --which tables
    ;;

  method)
    # E10 Method1 Bind-then-Act (news n=50 default)
    need_data
    MODELS="${MODELS:-qwen3vl internvl35}"
    N="${N_ART_METHOD:-50}"
    read -r -a M_ARR <<< "$MODELS"
    i=0
    for m in "${M_ARR[@]}"; do
      gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
      run_bg "$gpu" "method_${m}" scripts/smoke_bind_method1.py \
        --models "$m" --n "$N" --dest "$OUT/bind_method"
      i=$((i+1))
    done
    wait_all
    # ads variant optional
    if [[ "${RUN_ADS_METHOD:-0}" == "1" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python scripts/smoke_bind_ads.py \
        --models qwen3vl --n 50 --dest "$OUT/bind_ads" 2>&1 | tee "$LOG/method_ads.log"
    fi
    python scripts/score_all_tables.py --results-dir "$OUT" --which method
    ;;

  strict)
    need_data
    MODELS="${MODELS:-qwen3vl internvl35}"
    N="${N_STRICT:-25}"
    read -r -a M_ARR <<< "$MODELS"
    i=0
    for m in "${M_ARR[@]}"; do
      gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
      run_bg "$gpu" "strict_${m}" scripts/strict_small_eval.py \
        --models "$m" --n "$N" --dest "$OUT/strict_small"
      i=$((i+1))
    done
    wait_all
    python scripts/score_all_tables.py --results-dir "$OUT" --which strict
    ;;

  lang)
    need_data
    CUDA_VISIBLE_DEVICES="$GPUS" python scripts/run_lang_pairs.py \
      --gold "$GOLD" --dest-dir "$OUT/lang" \
      --models ${MODELS:-qwen3vl} --gpus "$GPUS" \
      --limit "${LIMIT:-0}" 2>&1 | tee "$LOG/lang.log"
    python scripts/score_all_tables.py --results-dir "$OUT" --which lang
    ;;

  score)
    python scripts/score_news_tables.py --dest-dir "$OUT/news_tables_act" || true
    python scripts/score_all_tables.py --results-dir "$OUT" --which all
    ;;

  all)
    "$0" e1
    "$0" tables
    "$0" method
    "$0" strict
    "$0" lang
    "$0" score
    ;;

  *)
    echo "Usage: $0 {smoke|e1|tables|method|strict|lang|score|all}"
    exit 1
    ;;
esac
echo "[done] mode=$MODE  results=$OUT  logs=$LOG"
