# Full-dataset setup (H100 / IMTP)

Smoke(3장)이 아니라 **벤치마크 전체**로 돌리는 방법입니다.

## 공개 상태 (2026-07)

| Dataset | Size | Access | Status |
|---|---|---|---|
| **VISTRA** | ~763 images × 4 langs (de/es/ru/zh) | Public GitHub | ✅ 자동 준비 가능 |
| **PRIM** | ~3k | HF gated `yztian/PRIM` | ✅ 토큰+약관 수락 후 |
| **IMTBench** | 2500 | 논문만 (아직 public dump 없음) | ⏳ 저자/릴리즈 대기 |
| **DIMT25** | 1k test | EULA + HF | ⏳ 이메일 신청 |
| **OCRMT30K** | ~30k / eval ~1.2k | 논문 릴리즈 | ⏳ 별도 |

---

## 1. VISTRA full (바로 가능)

```bash
cd /home/jovyan/CY/IMTP
git pull
cd experiments/iimt
source env.sh
export HF_TOKEN=hf_xxxx
export CUDA_VISIBLE_DEVICES=0

chmod +x scripts/data/download_full.sh
bash scripts/data/download_full.sh
# 또는 VISTRA만:
python scripts/data/prepare_vistra.py
```

생성 파일:

```text
data/manifests/vistra_en-de.jsonl
data/manifests/vistra_en-es.jsonl
data/manifests/vistra_en-ru.jsonl
data/manifests/vistra_en-zh.jsonl
data/manifests/vistra_all.jsonl
data/raw/vistra/{images,annotations}/
```

### B1 on VISTRA (언어별)

```bash
# German
BASELINES=B1 RUN_ID=vistra_b1_de \
  MANIFEST=data/manifests/vistra_en-de.jsonl \
  ./run_all.sh

# Chinese
BASELINES=B1 RUN_ID=vistra_b1_zh \
  MANIFEST=data/manifests/vistra_en-zh.jsonl \
  ./run_all.sh
```

Marian 모델 (언어별, 필요시):

```bash
# de
python scripts/run_baseline.py --baseline B1 \
  --manifest data/manifests/vistra_en-de.jsonl \
  --out_root outputs --run_id vistra_b1_de \
  --marian_model Helsinki-NLP/opus-mt-en-de

# zh (en→zh)
python scripts/run_baseline.py --baseline B1 \
  --manifest data/manifests/vistra_en-zh.jsonl \
  --out_root outputs --run_id vistra_b1_zh \
  --marian_model Helsinki-NLP/opus-mt-en-zh
```

**ETA (H100, B1 only, ~763 imgs):** ~20–45분 / language

---

## 2. PRIM full

```bash
# 1) https://huggingface.co/datasets/yztian/PRIM 에서 terms accept
# 2)
export HF_TOKEN=hf_xxxx
python scripts/data/prepare_prim.py \
  --tgt_langs ko,zh,ja

BASELINES=B1,B5 RUN_ID=prim001 \
  MANIFEST=data/manifests/prim_test.jsonl \
  ./run_all.sh
```

**ETA (H100, B1+B5, ~3k):** ~1–3시간

---

## 3. IMTBench / DIMT25 (아직 수동)

IMTBench public dump이 없으면:

1. 논문 authors에게 데이터 요청, 또는
2. 릴리즈되면 `data/raw/imtbench/`에 풀고  
   `scripts/data/prepare_imtbench.py` (추후)로 manifest 생성

DIMT25:

```text
EULA → dimt2025.contact@gmail.com
HF: liangyupu/DIMT2025.ICDAR.Track_2
```

---

## 4. Full run 추천 스케줄 (H100 1 GPU)

| Day | Job | ETA |
|---|---|---|
| 1 | VISTRA × B1 × {de,es,ru,zh} | ~2–3h |
| 1–2 | VISTRA × B5 × 4 langs | ~2–4h |
| 2 | PRIM × B1,B5 | ~1–3h |
| 3 | Stress A–E on VISTRA subsets | ~1–2h |
| later | IMTBench/DIMT when available | — |

병렬 (GPU 여러 장):

```bash
CUDA_VISIBLE_DEVICES=0 BASELINES=B1 RUN_ID=vistra_de MANIFEST=data/manifests/vistra_en-de.jsonl ./run_all.sh &
CUDA_VISIBLE_DEVICES=1 BASELINES=B1 RUN_ID=vistra_zh MANIFEST=data/manifests/vistra_en-zh.jsonl ./run_all.sh &
wait
```

---

## 5. 디스크 대략

| Data | Size |
|---|---|
| VISTRA | ~0.5–2 GB |
| PRIM | ~수 GB |
| HF models (Marian/VisTrans) | ~1–5 GB |

캐시는 `experiments/iimt/.cache/` 및 `data/raw/` 아래에 격리됩니다.
