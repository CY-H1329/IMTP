#!/usr/bin/env bash
set -euo pipefail
SN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SN_ROOT"
source .venv/bin/activate
source env.sh
export NEWS_DATA="${NEWS_DATA:-/home/gpuuser/chanyeong/data/hf_news_pack}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPUS="${GPUS:-0,1,2,3}"
if [[ -z "${HF_TOKEN:-}" && -f "$SN_ROOT/.hf_token" ]]; then
  export HF_TOKEN="$(tr -d '\n\r' < "$SN_ROOT/.hf_token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
LOGF="$SN_ROOT/logs/auto_pipeline.log"
exec >>"$LOGF" 2>&1

echo "[auto] start $(date -Is) token=$([[ -n ${HF_TOKEN:-} ]] && echo yes || echo no)"

wait_internvl() {
  echo "[auto] waiting for internvl35..."
  while pgrep -f "run_news_tables.py --models internvl35" >/dev/null 2>&1; do
    sleep 30
  done
  sleep 5
  echo "[auto] internvl done $(date -Is)"
}

clean_jsonl() {
  python3 - <<'PY'
import json
from pathlib import Path
for p in Path("results/news_tables_act").glob("*.jsonl"):
    good, bad = [], 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line=line.strip()
        if not line: continue
        try:
            json.loads(line); good.append(line)
        except Exception:
            bad += 1
    p.write_text("\n".join(good)+("\n" if good else ""), encoding="utf-8")
    if bad:
        print(f"[clean] {p.name} dropped {bad}")
PY
}

score_and_compare() {
  echo "[auto] scoring..."
  clean_jsonl
  python scripts/score_news_tables.py --dest-dir results/news_tables_act || true
  python scripts/score_all_tables.py --results-dir results --which tables || true
  python scripts/compare_from_tables_md.py results/news_tables_act/tables.md results/news_tables_act/compare_models.md || true
  python3 - <<'PY'
from pathlib import Path
text = Path("results/news_tables_act/tables.md").read_text()
rows = [ln for ln in text.splitlines() if ln.startswith("|") and ("Qwen" in ln or "InternVL" in ln or "gemma" in ln.lower() or "Gemma" in ln)]
Path("results/news_tables_act/COMPARE_NOTE.md").write_text(
    "# Compare note\n\nACT-item = all spans OK. 0% ACT != every span failed.\n\n"
    + "\n".join(rows[:30]) + "\n\nSee tables.md\n", encoding="utf-8")
print(Path("results/news_tables_act/COMPARE_NOTE.md").read_text())
PY
}

push_github() {
  echo "[auto] git commit + push..."
  cd /home/gpuuser/chanyeong/IMTP
  git add -f \
    experiments/selective_news/results/news_tables_act/ \
    experiments/selective_news/results/paper_tables/ \
    experiments/selective_news/run_4gpu.sh \
    experiments/selective_news/scripts/score_news_tables.py \
    experiments/selective_news/scripts/compare_from_tables_md.py \
    experiments/selective_news/run_auto_after_internvl.sh 2>/dev/null || true
  if ! git diff --cached --quiet; then
    export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-CY-H1329}"
    export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-CY-H1329@users.noreply.github.com}"
    export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
    export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
    git commit -m "Add InternVL tables results vs Qwen3-VL-8B; harness wait/score fixes." || true
  else
    echo "[auto] nothing to commit"
  fi
  git push origin HEAD:main && echo "[auto] push OK" || echo "[auto] WARN push failed"
  cd "$SN_ROOT"
}

free_space() {
  echo "[auto] freeing HF cache (chanyeong selective_news only)..."
  rm -rf .cache/huggingface/hub/models--OpenGVLab--InternVL3_5-8B-HF || true
  rm -rf .cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct || true
  rm -rf .cache/huggingface/hub/blobs .cache/huggingface/hub/.locks .cache/huggingface/xet || true
  mkdir -p .cache/huggingface/hub
  rm -f logs/*.pid || true
  df -h / | tail -1
}

run_next_full() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[auto] next=gemma12 WPG=2"
    export WORKERS_PER_GPU=2
    MODELS="gemma12 gemma12 gemma12 gemma12" ./run_4gpu.sh tables
    score_and_compare || true
    push_github || true
    free_space
    echo "[auto] next=qwen38 WPG=1"
    export WORKERS_PER_GPU=1
    MODELS="qwen38 qwen38 qwen38 qwen38" ./run_4gpu.sh tables
  else
    echo "[auto] next=qwen38 WPG=1 (no HF_TOKEN)"
    export WORKERS_PER_GPU=1
    MODELS="qwen38 qwen38 qwen38 qwen38" ./run_4gpu.sh tables
  fi
}

wait_internvl
score_and_compare
push_github
free_space
run_next_full
echo "[auto] DONE $(date -Is)"
