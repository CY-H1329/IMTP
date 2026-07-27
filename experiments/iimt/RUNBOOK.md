# IIMT 실험 Runbook (CY 워크스페이스)

**경로:** `/root/Desktop/workspace/CY/ReDesign/experiments/iimt`  
**공유 서버 규칙:** `.venv` + `env.sh` 만 사용. 전역 conda/pip/apt 수정 금지.

---

## 0. 공통 준비 (1회, ~30분)

```bash
cd /root/Desktop/workspace/CY/ReDesign/experiments/iimt

# 격리 venv 생성 (시스템 Python 건드리지 않음)
./setup.sh

# 매 세션마다
source env.sh

# GPU 확인 (다른 사용자 GPU 사용 중이면 번호 변경)
export CUDA_VISIBLE_DEVICES=0   # 예: 1, 2, ...
nvidia-smi
```

| 단계 | ETA |
|---|---|
| `./setup.sh` (venv + core deps) | **10–20분** |
| baseline별 추가 pip (paddleocr 등) | **10–30분** |
| third_party clone + HF weights | **30분–3시간** |

---

## 1. Baseline 실험 (B1–B7 + Ours)

모든 실험은 동일 패턴:

```bash
cd /root/Desktop/workspace/CY/ReDesign/experiments/iimt
source env.sh

python scripts/run_baseline.py \
  --baseline <ID> \
  --manifest data/manifests/<manifest>.jsonl \
  --out_root outputs \
  --run_id <run_id>
```

---

### B1 — PaddleOCR + MarianMT (Cascade)

| 항목 | 내용 |
|---|---|
| **목적** | OCR → MT → 렌더 전형 cascade. error propagation baseline |
| **필요** | `pip install paddleocr paddlepaddle-gpu transformers torch` (venv 내부) |
| **명령** | 아래 |
| **VRAM** | ~4–8 GB |

```bash
source env.sh
python scripts/run_baseline.py \
  --baseline B1 \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs --run_id b1_smoke

# IMTBench (manifest 준비 후)
python scripts/run_baseline.py \
  --baseline B1 \
  --manifest data/manifests/imtbench_test.jsonl \
  --out_root outputs --run_id b1_imtbench_en-ko \
  --marian_model Helsinki-NLP/opus-mt-en-ko
```

| 데이터 규모 | ETA |
|---|---|
| smoke (3장) | **~1–2분** |
| VISTRA (772) | **~20–40분** |
| IMTBench (2,500) | **~1–2시간** |

---

### B2 — Tesseract + mBART (Cascade ablation)

| 항목 | 내용 |
|---|---|
| **목적** | OCR 엔진 ablation (Tesseract vs Paddle) |
| **필요** | `tesseract` (시스템), `pip install pytesseract transformers` |
| **VRAM** | ~6–10 GB |

```bash
source env.sh
python scripts/run_baseline.py \
  --baseline B2 \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs --run_id b2_smoke
```

| 데이터 규모 | ETA |
|---|---|
| smoke (3) | **~2–3분** |
| IMTBench (2,500) | **~1.5–2.5시간** |

---

### B3 — AnyTrans-open

| 항목 | 내용 |
|---|---|
| **목적** | OCR + 오픈 LLM + diffusion/edit. 문맥 번역 baseline |
| **필요** | `ReDesign/third_party/AnyTrans` clone |
| **VRAM** | ~16–20 GB (Qwen-7B 기준) |

```bash
# clone (1회)
cd /root/Desktop/workspace/CY/ReDesign/third_party
git clone https://github.com/qzp2018/AnyTrans.git

cd /root/Desktop/workspace/CY/ReDesign/experiments/iimt
source env.sh
python scripts/run_baseline.py \
  --baseline B3 \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs --run_id b3_smoke
```

| 데이터 규모 | ETA |
|---|---|
| smoke (3) | **~2–5분** |
| IMTBench (2,500) | **~4–10시간** |

---

### B4 — Translatotron-V (End-to-end IIMT)

| 항목 | 내용 |
|---|---|
| **목적** | 이미지→이미지 end-to-end. Structure-BLEU 비교 |
| **필요** | `third_party/Translatotron-V` + Google Drive checkpoint |
| **VRAM** | ~12–16 GB |

```bash
cd /root/Desktop/workspace/CY/ReDesign/third_party
git clone https://github.com/DeepLearnXMU/translatotron-v Translatotron-V

cd /root/Desktop/workspace/CY/ReDesign/experiments/iimt
source env.sh
python scripts/run_baseline.py \
  --baseline B4 \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs --run_id b4_smoke \
  --checkpoint /path/to/translatotron_v.ckpt
```

| 데이터 규모 | ETA |
|---|---|
| smoke (3) | **~1–3분** |
| IMTBench (2,500) | **~1.5–3.5시간** |

---

### B5 — VisTrans / PRIM

| 항목 | 내용 |
|---|---|
| **목적** | 실사용 poster/product 강한 practical IIMT baseline |
| **필요** | HF `yztian/VisTrans`, PRIM dataset (gated) |
| **VRAM** | ~10–14 GB |

```bash
source env.sh
python scripts/run_baseline.py \
  --baseline B5 \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs --run_id b5_smoke \
  --hf_model yztian/VisTrans
```

| 데이터 규모 | ETA |
|---|---|
| smoke (3) | **~1–2분** |
| PRIM (~3k) | **~45분–2시간** |
| IMTBench (2,500) | **~45분–2시간** |

---

### B6 — InImageTrans-open (MLLM)

| 항목 | 내용 |
|---|---|
| **목적** | hallucination / omission 분석용 오픈 VLM |
| **필요** | `third_party/InImageTrans` 또는 Qwen2-VL |
| **VRAM** | ~16–20 GB |

```bash
source env.sh
python scripts/run_baseline.py \
  --baseline B6 \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs --run_id b6_smoke
```

| 데이터 규모 | ETA |
|---|---|
| smoke (3) | **~2–5분** |
| IMTBench (2,500) | **~4–14시간** |

---

### B7 — DIMT25 Track2 (Document)

| 항목 | 내용 |
|---|---|
| **목적** | 복잡 문서 레이아웃 en→zh |
| **필요** | DIMT25 EULA + HF `liangyupu/DIMT2025.ICDAR.Track_2` |
| **VRAM** | ~14–20 GB |

```bash
source env.sh
python scripts/run_baseline.py \
  --baseline B7 \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs --run_id b7_smoke
```

| 데이터 규모 | ETA |
|---|---|
| DIMT25 test (1k) | **~1–3시간** |

---

### Ours — ReDesign text pipeline

| 항목 | 내용 |
|---|---|
| **목적** | 본 연구 방법 (parse → edit → re-render) |
| **필요** | ReDesign GPU pipeline + weights |
| **VRAM** | 프로젝트 설정에 따름 |

```bash
source env.sh
python scripts/run_baseline.py \
  --baseline Ours \
  --manifest data/manifests/smoke_test.jsonl \
  --out_root outputs --run_id ours_smoke \
  --tool_gpus 0
```

| 데이터 규모 | ETA |
|---|---|
| smoke (3) | **~3–10분** (파이프라인에 따라) |
| IMTBench (2,500) | **~2–8시간** |

---

## 2. 전체 비교 한 번에 (`run_all.sh`)

```bash
cd /root/Desktop/workspace/CY/ReDesign/experiments/iimt
source env.sh

# smoke (3장, 빠른 확인)
./run_all.sh

# 커스텀
RUN_ID=imtbench001 \
MANIFEST=data/manifests/imtbench_test.jsonl \
BASELINES=B1,B5,Ours \
./run_all.sh
```

| 설정 | ETA |
|---|---|
| smoke × 8 baselines (3장) | **~15–30분** |
| IMTBench × 8 baselines (2,500장) | **~15–40시간** (GPU 1대) |

---

## 3. 평가 (추론 후)

`run_all.sh`에 포함되거나 개별 실행:

```bash
source env.sh

python scripts/eval/compute_metrics.py \
  --pred_root outputs/run001 \
  --ref_manifest data/manifests/smoke_test.jsonl \
  --out_csv results/run001_per_sample.csv \
  --run_id run001

python scripts/eval/aggregate_results.py \
  --in_csv results/run001_per_sample.csv \
  --out_csv results/run001_aggregate.csv

python scripts/eval/error_decomposition.py \
  --pred_root outputs/run001 \
  --ref_manifest data/manifests/smoke_test.jsonl \
  --out_csv results/run001_error_breakdown.csv \
  --run_id run001
```

| 작업 | ETA |
|---|---|
| metrics (smoke 3) | **~10초** |
| metrics (IMTBench 2,500) | **~10–20분** |
| COMET 추가 시 | **+30–60분** |

---

## 4. Stress Test (A–E)

```bash
source env.sh

# manifest 생성
python scripts/stress/make_stress_manifests.py \
  --base_manifest data/manifests/smoke_test.jsonl \
  --out_dir data/manifests/stress

# A: OCR noise
RUN_ID=stressA MANIFEST=data/manifests/stress/stress_A_ocr_noise.jsonl \
  BASELINES=B1 ./run_all.sh

# B: overflow
RUN_ID=stressB MANIFEST=data/manifests/stress/stress_B_overflow.jsonl \
  BASELINES=B1,B5 ./run_all.sh

# C: script transfer
RUN_ID=stressC MANIFEST=data/manifests/stress/stress_C_script.jsonl \
  BASELINES=B1,B5 ./run_all.sh

# D: real-world poster/product
RUN_ID=stressD MANIFEST=data/manifests/stress/stress_D_realworld.jsonl \
  BASELINES=B5,Ours ./run_all.sh

# E: re-OCR consistency
RUN_ID=stressE MANIFEST=data/manifests/stress/stress_E_reocr.jsonl \
  BASELINES=B1,B4,B5 ./run_all.sh
```

| Test | 목적 | ETA (smoke) | ETA (IMTBench subset) |
|---|---|---|---|
| A OCR noise | cascade error propagation | **~2분** | **~1–2h** (B1 재실행) |
| B overflow | 긴 target lang overflow | **~2분** | **~30–60% of full** |
| C script | en→ko/ja/ar 일반화 | **~2분** | lang당 **~2–4h** |
| D real-world | poster/product | **~2분** | **~1–2h** |
| E re-OCR | 출력 텍스트 무결성 | **~1분** | **~20–40분** |

---

## 5. 벤치마크별 manifest

| Bench | Manifest (현재) | 전체 규모 | 준비 |
|---|---|---|---|
| smoke | `data/manifests/smoke_test.jsonl` | 3 | ✅ 포함 |
| vistra | `data/manifests/vistra_test.jsonl` | 772 (full) | clone + manifest 작성 |
| imtbench | `data/manifests/imtbench_test.jsonl` | 2,500 | 논문 release |
| prim | (작성 필요) | ~3k | HF gated |
| dimt25 | (작성 필요) | 1k test | EULA |

---

## 6. 공유 서버 체크리스트

- [ ] `source env.sh` 로 **로컬 .venv** 만 활성화
- [ ] `CUDA_VISIBLE_DEVICES` 로 GPU 명시 (다른 사용자 GPU 사용 금지)
- [ ] `pip install` 은 **반드시 venv 활성화 후**
- [ ] `conda activate` / `pip install --user` / `sudo apt` **사용 금지**
- [ ] HF 캐시는 `$IIMT_ROOT/.cache/huggingface` (워크스pace 격리)
- [ ] 출력은 `outputs/`, `results/` 에만 저장

---

## 7. 전체 일정 ETA (GPU 1대, IMTBench en→ko)

| Phase | 내용 | ETA |
|---|---|---|
| 0 | setup + smoke | **0.5일** |
| 1 | B1,B5,Ours on IMTBench | **~1–2일** |
| 2 | B2–B4,B6,B7 | **~3–5일** |
| 3 | VISTRA + PRIM | **~2–3일** |
| 4 | Stress A–E | **~1–2일** |
| **Total** | | **~7–12일** |

GPU 2–3대 병렬 시 **~3–5일**로 단축 가능.

---

프로토콜 상세: `../../docs/EXPERIMENT_PROTOCOL_OPENSOURCE_IIMT.md`
