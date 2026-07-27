#!/usr/bin/env bash
# Run full open-source IIMT comparison (CY workspace, isolated venv)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Load isolated env (does NOT touch system conda/pip)
# shellcheck disable=SC1091
source "$ROOT/env.sh"

RUN_ID="${RUN_ID:-run001}"
MANIFEST="${MANIFEST:-data/manifests/smoke_test.jsonl}"
BASELINES="${BASELINES:-B1,B2,B3,B4,B5,B6,B7,Ours}"
OUT_ROOT="${OUT_ROOT:-outputs}"

IFS=',' read -ra BL_LIST <<< "$BASELINES"
for BL in "${BL_LIST[@]}"; do
  echo "========== $BL =========="
  python scripts/run_baseline.py \
    --baseline "$BL" \
    --manifest "$MANIFEST" \
    --out_root "$OUT_ROOT" \
    --run_id "$RUN_ID"
done

RESULTS="results/${RUN_ID}"
mkdir -p "$RESULTS"

python scripts/eval/compute_metrics.py \
  --pred_root "$OUT_ROOT/$RUN_ID" \
  --ref_manifest "$MANIFEST" \
  --out_csv "$RESULTS/${RUN_ID}_per_sample.csv" \
  --run_id "$RUN_ID"

python scripts/eval/aggregate_results.py \
  --in_csv "$RESULTS/${RUN_ID}_per_sample.csv" \
  --out_csv "$RESULTS/${RUN_ID}_aggregate.csv"

python scripts/eval/error_decomposition.py \
  --pred_root "$OUT_ROOT/$RUN_ID" \
  --ref_manifest "$MANIFEST" \
  --out_csv "$RESULTS/${RUN_ID}_error_breakdown.csv" \
  --run_id "$RUN_ID"

echo ""
echo "Done. Results in $RESULTS/"
