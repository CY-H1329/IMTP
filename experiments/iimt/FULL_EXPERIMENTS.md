# IMTP Full Experiments — 설명 + Command

경로: `experiments/iimt/`  
Repo: https://github.com/CY-H1329/IMTP  
서버: H100 JupyterLab (`/home/jovyan/CY/IMTP`)

---

## 0. 매 세션 시작

```bash
cd /home/jovyan/CY/IMTP && git pull
cd experiments/iimt
source env.sh
export CUDA_VISIBLE_DEVICES=0          # nvidia-smi로 빈 GPU 확인
export HF_TOKEN=hf_xxxxxxxx            # Hub 다운로드용
```

| 항목 | 설명 |
|---|---|
| `source env.sh` | 로컬 `.venv` + 캐시 격리 (공유 서버 안전) |
| `CUDA_VISIBLE_DEVICES` | 사용할 GPU만 지정 |
| `HF_TOKEN` | Marian/VisTrans/PRIM 등 HF 접근 |

GPU 확인:

```bash
python - <<'PY'
import torch
print("torch.cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
# 실행 중: watch -n 1 nvidia-smi
```

---

## 1. 데이터 준비 (Full)

### 1.1 한 번에

```bash
chmod +x scripts/data/download_full.sh
bash scripts/data/download_full.sh
```

### 1.2 VISTRA only (~763 images × 4 langs)

**무엇을 테스트:** 자연 이미지 scene text의 시각 맥락 번역. OCR 오류 vs 번역 오류 분리 분석에 최적.

```bash
python scripts/data/prepare_vistra.py
# → data/manifests/vistra_en-{de,es,ru,zh}.jsonl
# → data/raw/vistra/
```

### 1.3 PRIM (~3k, gated)

**무엇을 테스트:** 실촬영 한 줄 텍스트, 복잡 배경/폰트. practical IIMT.

```bash
# https://huggingface.co/datasets/yztian/PRIM 약관 수락 후
python scripts/data/prepare_prim.py --tgt_langs ko,zh,ja
# → data/manifests/prim_test.jsonl
```

### 1.4 아직 수동

| Dataset | 규모 | 비고 |
|---|---|---|
| IMTBench | 2500 | 논문만, public dump 대기 |
| DIMT25 | 1k test | EULA 필요 |
| OCRMT30K | ~1.2k eval | 논문 릴리즈 |

---

## 2. Main Baselines (무엇을 / 왜 / Command)

공통 패턴:

```bash
python scripts/run_baseline.py \
  --baseline <ID> \
  --manifest <MANIFEST.jsonl> \
  --out_root outputs \
  --run_id <RUN_ID>
```

또는:

```bash
BASELINES=<ID> RUN_ID=<RUN_ID> MANIFEST=<path> ./run_all.sh
```

---

### B1 — PaddleOCR + MarianMT (Cascade)

| | |
|---|---|
| **무엇을** | OCR → Marian 번역 → 박스에 텍스트 렌더 |
| **왜** | 전형적 cascade. OCR 오류 전파·overflow baseline |
| **GPU** | Marian → CUDA / PaddleOCR → GPU(가능 시) |
| **ETA smoke** | ~3–10분 |
| **ETA VISTRA(~763)** | ~20–45분/lang |
| **ETA PRIM(~3k)** | ~1–2시간 |

```bash
# smoke
python scripts/run_baseline.py --baseline B1 \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs --run_id smoke \
  --marian_model Helsinki-NLP/opus-mt-tc-big-en-ko

# VISTRA full — German
BASELINES=B1 RUN_ID=vistra_b1_de \
  MANIFEST=data/manifests/vistra_en-de.jsonl ./run_all.sh

# VISTRA — Chinese
BASELINES=B1 RUN_ID=vistra_b1_zh \
  MANIFEST=data/manifests/vistra_en-zh.jsonl ./run_all.sh

# VISTRA — Spanish / Russian
BASELINES=B1 RUN_ID=vistra_b1_es MANIFEST=data/manifests/vistra_en-es.jsonl ./run_all.sh
BASELINES=B1 RUN_ID=vistra_b1_ru MANIFEST=data/manifests/vistra_en-ru.jsonl ./run_all.sh
```

로그에 `[B1] torch device = cuda` / `Marian ready on cuda` 가 보여야 GPU 사용.

---

### B2 — Tesseract + mBART

| | |
|---|---|
| **무엇을** | Tesseract OCR + mBART 다국어 MT |
| **왜** | OCR 엔진 ablation (Paddle vs Tesseract) |
| **필요** | `conda install -y -c conda-forge tesseract` + `pip install pytesseract` |
| **ETA VISTRA** | ~30–60분/lang |

```bash
conda install -y -c conda-forge tesseract
which tesseract

BASELINES=B2 RUN_ID=vistra_b2_de \
  MANIFEST=data/manifests/vistra_en-de.jsonl ./run_all.sh
```

tesseract 없으면 `meta.json` → `setup_required` (crash 없음).

---

### B3 — AnyTrans

| | |
|---|---|
| **무엇을** | OCR + LLM 문맥 번역 + diffusion/edit |
| **왜** | 문맥 인식 edit baseline (cascade보다 semantic) |
| **필요** | `third_party/AnyTrans` |
| **ETA VISTRA** | ~2–6시간/lang |

```bash
mkdir -p ../../third_party && cd ../../third_party
git clone https://github.com/qzp2018/AnyTrans.git
cd -

BASELINES=B3 RUN_ID=vistra_b3_de \
  MANIFEST=data/manifests/vistra_en-de.jsonl ./run_all.sh
```

---

### B4 — Translatotron-V

| | |
|---|---|
| **무엇을** | End-to-end 이미지→이미지 IIMT |
| **왜** | OCR/MT 분리 없는 E2E + Structure-BLEU 비교 |
| **필요** | repo + checkpoint |
| **ETA VISTRA** | ~1–2.5시간/lang |

```bash
cd ../../third_party
git clone https://github.com/DeepLearnXMU/translatotron-v Translatotron-V
# checkpoint 준비 후

cd -
python scripts/run_baseline.py --baseline B4 \
  --manifest data/manifests/vistra_en-de.jsonl \
  --out_root outputs --run_id vistra_b4_de \
  --checkpoint /path/to/translatotron_v.ckpt
```

---

### B5 — VisTrans / PRIM

| | |
|---|---|
| **무엇을** | visual text / background 분리 practical IIMT |
| **왜** | 실사용 poster/product strong open baseline |
| **필요** | HF `yztian/VisTrans` |
| **ETA VISTRA** | ~20–60분/lang |
| **ETA PRIM** | ~45분–2시간 |

```bash
BASELINES=B5 RUN_ID=vistra_b5_de \
  MANIFEST=data/manifests/vistra_en-de.jsonl ./run_all.sh

BASELINES=B1,B5 RUN_ID=prim_b1b5 \
  MANIFEST=data/manifests/prim_test.jsonl ./run_all.sh
```

---

### B6 — InImageTrans-open

| | |
|---|---|
| **무엇을** | 오픈 VLM으로 번역/편집 + omission 분석 |
| **왜** | hallucination / omission / repetition |
| **ETA VISTRA** | ~2–8시간/lang |

```bash
BASELINES=B6 RUN_ID=vistra_b6_de \
  MANIFEST=data/manifests/vistra_en-de.jsonl ./run_all.sh
```

---

### B7 — DIMT25 Track2

| | |
|---|---|
| **무엇을** | 문서 OCR-free IIMT (en→zh) |
| **왜** | 복잡 레이아웃 문서 트랙 |
| **필요** | DIMT25 EULA + HF |
| **ETA 1k** | ~45–90분 |

```bash
# 데이터 준비 후
BASELINES=B7 RUN_ID=dimt_b7 \
  MANIFEST=data/manifests/dimt25_test.jsonl ./run_all.sh
```

---

### Ours — ReDesign

| | |
|---|---|
| **무엇을** | parse → edit → re-render (본 연구) |
| **왜** | B1–B7과 동일 조건 비교 |
| **ETA VISTRA** | ~1–4시간/lang |

```bash
python scripts/run_baseline.py --baseline Ours \
  --manifest data/manifests/vistra_en-de.jsonl \
  --out_root outputs --run_id vistra_ours_de \
  --tool_gpus 0
```

---

## 3. Stress Tests A–E (무엇을 / 왜 / Command)

먼저 subset 생성:

```bash
# smoke 기반 (빠른 검증)
python scripts/stress/make_stress_manifests.py \
  --base_manifest data/manifests/smoke_test.jsonl \
  --out_dir data/manifests/stress

# full VISTRA 기반 (본실험)
python scripts/stress/make_stress_manifests.py \
  --base_manifest data/manifests/vistra_en-de.jsonl \
  --out_dir data/manifests/stress_vistra_de
```

---

### Test A — OCR-noise robustness

| | |
|---|---|
| **무엇을** | OCR 텍스트를 의도적으로 corrupt |
| **왜** | cascade가 OCR 오류에 얼마나 무너지는지 (error propagation) |
| **기대** | B1 급락, E2E(B4/B5)는 상대적으로 덜 민감할 수 있음 |

```bash
RUN_ID=stressA_b1 MANIFEST=data/manifests/stress/stress_A_ocr_noise.jsonl \
  BASELINES=B1 ./run_all.sh
```

---

### Test B — Layout overflow

| | |
|---|---|
| **무엇을** | target 문장이 source보다 길 때 truncation/overflow |
| **왜** | typography adaptation / 박스 안에 글자 유지 능력 |
| **기대** | cascade·단순 렌더에서 overflow↑ |

```bash
RUN_ID=stressB_b1 MANIFEST=data/manifests/stress/stress_B_overflow.jsonl \
  BASELINES=B1 ./run_all.sh
```

---

### Test C — Script transfer

| | |
|---|---|
| **무엇을** | en→ko / ja / zh / ar 등 script 난이도 |
| **왜** | multilingual / script generalization |
| **기대** | Latin 외 script에서 CER·렌더 품질 저하 |

```bash
RUN_ID=stressC_b1 MANIFEST=data/manifests/stress/stress_C_script.jsonl \
  BASELINES=B1 ./run_all.sh
```

---

### Test D — Real-world poster/product

| | |
|---|---|
| **무엇을** | 광고·포스터·패키지형 이미지 |
| **왜** | 실사용 가능성 (PRIM/VisTrans 강점 영역) |
| **기대** | B5 / Ours > cascade |

```bash
RUN_ID=stressD_b1b5 MANIFEST=data/manifests/stress/stress_D_realworld.jsonl \
  BASELINES=B1,B5 ./run_all.sh
```

---

### Test E — Re-OCR consistency

| | |
|---|---|
| **무엇을** | 생성 이미지에 OCR 재실행 → target과 일치? |
| **왜** | final image의 text integrity (번역은 맞는데 글자가 깨진 경우 검출) |
| **기대** | 높은 exact match / 낮은 CER |

```bash
RUN_ID=stressE_b1 MANIFEST=data/manifests/stress/stress_E_reocr.jsonl \
  BASELINES=B1 ./run_all.sh
```

---

## 4. 평가만 다시

```bash
python scripts/eval/compute_metrics.py \
  --pred_root outputs/<RUN_ID> \
  --ref_manifest data/manifests/<manifest>.jsonl \
  --out_csv results/<RUN_ID>_per_sample.csv \
  --run_id <RUN_ID>

python scripts/eval/aggregate_results.py \
  --in_csv results/<RUN_ID>_per_sample.csv \
  --out_csv results/<RUN_ID>_aggregate.csv

python scripts/eval/error_decomposition.py \
  --pred_root outputs/<RUN_ID> \
  --ref_manifest data/manifests/<manifest>.jsonl \
  --out_csv results/<RUN_ID>_errors.csv \
  --run_id <RUN_ID>
```

| 지표 | 의미 |
|---|---|
| BLEU / chrF / COMET | 번역 품질 |
| CER / WER / exact match | 생성 이미지 re-OCR 텍스트 정확도 |
| SSIM 등 | 시각 유사도 (가능 시) |

---

## 5. 추천 Full 스케줄 (H100 1 GPU)

| 순서 | 실험 | Command 요약 | ETA |
|---|---|---|---|
| 1 | 데이터 | `bash scripts/data/download_full.sh` | ~10–30분 |
| 2 | B1 × VISTRA × 4 langs | `run_all.sh` ×4 | ~2–3시간 |
| 3 | B5 × VISTRA × 1–4 langs | `BASELINES=B5 ...` | ~1–4시간 |
| 4 | B1+B5 × PRIM | `prim_test.jsonl` | ~1–3시간 |
| 5 | Stress A–E (VISTRA-de) | stress manifests | ~1–2시간 |
| 6 | B2/B3/Ours (여유 시) | 개별 | 가변 |

GPU 병렬 예:

```bash
CUDA_VISIBLE_DEVICES=0 BASELINES=B1 RUN_ID=vistra_de MANIFEST=data/manifests/vistra_en-de.jsonl ./run_all.sh &
CUDA_VISIBLE_DEVICES=1 BASELINES=B1 RUN_ID=vistra_zh MANIFEST=data/manifests/vistra_en-zh.jsonl ./run_all.sh &
wait
```

---

## 6. 결과 위치

```text
outputs/<RUN_ID>/<BASELINE>/<benchmark>/<sample_id>/
  pred_text.txt
  pred_image.png
  reocr.json
  meta.json          # status, latency, device 관련 info

results/<RUN_ID>_*.csv
```

---

## 7. 지금 바로 복붙 (Full 시작)

```bash
cd /home/jovyan/CY/IMTP && git pull
cd experiments/iimt
source env.sh
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=hf_xxxx

bash scripts/data/download_full.sh

BASELINES=B1 RUN_ID=vistra_b1_de \
  MANIFEST=data/manifests/vistra_en-de.jsonl \
  ./run_all.sh
```

성공 확인:

```bash
# GPU 사용 로그
# [B1] torch device = cuda

ls outputs/vistra_b1_de/B1/VISTRA/ | wc -l
head results/vistra_b1_de_aggregate.csv
```
