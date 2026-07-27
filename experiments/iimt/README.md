# IIMT Open-Source Experiments

실험 실행 인프라. 프로토콜 상세: `../../docs/EXPERIMENT_PROTOCOL_OPENSOURCE_IIMT.md`

## Quick start

```bash
cd experiments/iimt
chmod +x setup.sh run_all.sh
./setup.sh

# Smoke test (3 sample images)
./run_all.sh

# Custom run
RUN_ID=run002 MANIFEST=data/manifests/imtbench_test.jsonl BASELINES=B1,Ours ./run_all.sh
```

## Directory layout

```
experiments/iimt/
├── config/baselines.yaml      # baseline IDs & default models
├── data/manifests/*.jsonl     # input samples (one JSON per line)
├── scripts/
│   ├── run_baseline.py        # unified runner
│   ├── baselines/             # B1–B7 + Ours
│   ├── eval/                  # metrics & error breakdown
│   └── stress/                # stress-test manifest builder
├── outputs/{run_id}/{baseline}/{benchmark}/{sample_id}/
│   ├── pred_text.txt
│   ├── pred_image.png
│   ├── reocr.json
│   └── meta.json
└── results/{run_id}_*.csv
```

## Run one baseline

```bash
source .venv/bin/activate

# B1: PaddleOCR + MarianMT
python scripts/run_baseline.py \
  --baseline B1 \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs \
  --run_id run001

# Ours: ReDesign
python scripts/run_baseline.py \
  --baseline Ours \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs \
  --run_id run001 \
  --tool_gpus 0
```

## Stress tests

```bash
python scripts/stress/make_stress_manifests.py \
  --base_manifest data/manifests/smoke_test.jsonl \
  --out_dir data/manifests/stress

RUN_ID=stress001 MANIFEST=data/manifests/stress/stress_A_ocr_noise.jsonl ./run_all.sh
```

## Third-party setup

Clone into `ReDesign/third_party/`:

```bash
cd ../../third_party
git clone https://github.com/qzp2018/AnyTrans.git
# Translatotron-V, InImageTrans: follow respective papers
```

HF models (B5, B7) download automatically on first run if `transformers` is installed.

## Evaluate only

```bash
python scripts/eval/compute_metrics.py \
  --pred_root outputs/run001 \
  --ref_manifest data/manifests/smoke_test.jsonl \
  --out_csv results/run001_per_sample.csv

python scripts/eval/aggregate_results.py \
  --in_csv results/run001_per_sample.csv \
  --out_csv results/run001_aggregate.csv
```

## Baseline status

| ID | Script | Needs |
|----|--------|-------|
| B1 | cascade_paddle_marian.py | paddleocr, transformers |
| B2 | cascade_tesseract_mbart.py | tesseract, pytesseract, transformers |
| B3 | anytrans_infer.py | third_party/AnyTrans |
| B4 | translatotron_v_infer.py | third_party/Translatotron-V |
| B5 | vistrans_infer.py | transformers, GPU |
| B6 | inimagetrans_infer.py | third_party/InImageTrans |
| B7 | dimt25_infer.py | transformers, GPU |
| Ours | redesign_infer.py | ReDesign GPU pipeline + weights |

Baselines without dependencies write `meta.json` with `status: setup_required` instead of crashing.
