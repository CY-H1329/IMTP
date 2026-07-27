#!/usr/bin/env bash
# Prepare full open datasets for IIMT experiments (H100 / CY IMTP)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env.sh" 2>/dev/null || true

echo "==> [1/3] VISTRA (public, ~763 images × 4 langs)"
python scripts/data/prepare_vistra.py \
  --data_root data/raw/vistra \
  --out_manifest_dir data/manifests

echo ""
echo "==> [2/3] PRIM (HF gated — needs HF_TOKEN + accept terms)"
if [[ -n "${HF_TOKEN:-}" ]]; then
  python scripts/data/prepare_prim.py \
    --data_root data/raw/prim \
    --out_manifest_dir data/manifests \
    --tgt_langs ko,zh,ja || echo "[warn] PRIM failed — accept https://huggingface.co/datasets/yztian/PRIM"
else
  echo "[skip] set HF_TOKEN to download PRIM"
fi

echo ""
echo "==> [3/3] IMTBench / DIMT25 / OCRMT30K"
echo "  IMTBench: not publicly hosted yet (arxiv:2603.10495) — contact authors / watch paper repo"
echo "  DIMT25:   EULA → dimt2025.contact@gmail.com + HF Track1/2"
echo "  OCRMT30K: follow Lan et al. release / Translatotron-V data links"

echo ""
echo "Ready manifests:"
ls -lh data/manifests/vistra_*.jsonl 2>/dev/null || true
ls -lh data/manifests/prim_*.jsonl 2>/dev/null || true

echo ""
echo "Example full runs:"
echo "  BASELINES=B1 RUN_ID=vistra_de MANIFEST=data/manifests/vistra_en-de.jsonl ./run_all.sh"
echo "  BASELINES=B1,B5 RUN_ID=prim_ko MANIFEST=data/manifests/prim_test.jsonl ./run_all.sh"
