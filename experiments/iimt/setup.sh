#!/usr/bin/env bash
# CY IIMT 실험 환경 설정 — 공유 서버 안전 모드
#
# ✅ 하는 것:  experiments/iimt/.venv 생성 + requirements 설치
# ❌ 하지 않음: conda 전역 env 수정, pip install --user, apt, git config 변경
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
echo "==> [CY IIMT] Creating isolated venv at $ROOT/.venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

mkdir -p .cache/{huggingface,torch,pip,tmp}
mkdir -p outputs results data/manifests/stress ../../third_party

echo ""
echo "==> Setup complete (isolated, no system changes)"
echo ""
echo "Next steps:"
echo "  cd $ROOT"
echo "  source env.sh"
echo ""
echo "Smoke test:"
echo "  python scripts/run_baseline.py --baseline B1 \\"
echo "    --manifest data/manifests/smoke_test.jsonl \\"
echo "    --out_root outputs --run_id smoke"
echo ""
echo "Optional baseline deps (install INSIDE .venv only when needed):"
echo "  pip install paddleocr paddlepaddle-gpu transformers torch"
echo "  pip install pytesseract"
echo ""
echo "Third-party (clone to ReDesign/third_party/, NOT system-wide):"
echo "  cd $ROOT/../../third_party"
echo "  git clone https://github.com/qzp2018/AnyTrans.git"
echo "  git clone https://github.com/DeepLearnXMU/translatotron-v Translatotron-V"
echo "  git clone https://github.com/BITHLP/PRIM"
