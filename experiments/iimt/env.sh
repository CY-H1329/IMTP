#!/usr/bin/env bash
# CY IIMT 실험 전용 환경 — 공유 서버 안전 모드
#
# 사용법 (매 터미널 세션):
#   cd /root/Desktop/workspace/CY/ReDesign/experiments/iimt
#   source env.sh
#
# 주의:
#   - 시스템 Python / 전역 pip / conda base 를 건드리지 않습니다.
#   - .venv 가 없으면 ./setup.sh 를 먼저 실행하세요.
#   - GPU는 CUDA_VISIBLE_DEVICES 로 명시적으로 지정하세요.

_IIMT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 격리: 캐시/임시 파일을 CY 워크스페이스 내부로 ----
export IIMT_ROOT="$_IIMT_ROOT"
export HF_HOME="${HF_HOME:-$_IIMT_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$_IIMT_ROOT/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$_IIMT_ROOT/.cache/pip}"
export TMPDIR="${TMPDIR:-$_IIMT_ROOT/.cache/tmp}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# ---- GPU: 기본 0번, 다른 사용자 GPU 침범 방지를 위해 반드시 확인 ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---- venv 활성화 (로컬만) ----
if [[ -f "$_IIMT_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$_IIMT_ROOT/.venv/bin/activate"
else
  echo "[WARN] .venv 없음 → ./setup.sh 실행 후 다시 source env.sh"
fi

export PYTHONPATH="$_IIMT_ROOT/scripts:${PYTHONPATH:-}"

# ---- run helpers ----
iimt_run_id() {
  local bl="$1" bench="$2" pair="$3"
  date +%Y%m%d_%H%M_"${bl}_${bench}_${pair}"
}

iimt_activate() {
  cd "$_IIMT_ROOT" || return 1
  echo "IIMT_ROOT=$_IIMT_ROOT"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "Python: $(which python 2>/dev/null || echo 'not found')"
}

iimt_activate
