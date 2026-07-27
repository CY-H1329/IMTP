# Open-Source Only: Multilingual In-Image Rewriting — Experimental Protocol

**Version:** 1.0  
**Scope:** Closed/proprietary models excluded. All baselines must be reproducible from public code and weights.

---

## 1. Objective

We study **open-source multilingual in-image rewriting (IIMT)**: given an image containing source-language text, generate an edited image that preserves meaning, layout, typography, and background fidelity while translating text into a target language.

We compare three families of approaches:

1. **OCR + MT cascade** — modular, interpretable, error-prone under OCR noise.
2. **End-to-end IIMT** — joint translation and rendering, stronger grounding but weaker typography control.
3. **Practical multilingual rewriting** — layout-aware systems optimized for real posters, documents, and product packaging.

**Central hypothesis.** Cascaded systems fail mainly under OCR noise and layout overflow; end-to-end systems fail under grounding and typography mismatch; practical multilingual systems (VisTrans/PRIM, DIMT25) perform best on real-world poster/product cases.

---

## 2. Baselines (open-source only)

| ID | Category | System | Public artifact | Role |
|----|----------|--------|-----------------|------|
| B1 | OCR+MT cascade | PaddleOCR + MarianMT | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [MarianMT](https://huggingface.co/Helsinki-NLP) | Strong modular baseline |
| B2 | OCR+MT cascade | Tesseract + mBART | [Tesseract](https://github.com/tesseract-ocr/tesseract), [mBART](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) | Legacy cascade baseline |
| B3 | Edit-based | **AnyTrans** | [github.com/qzp2018/AnyTrans](https://github.com/qzp2018/AnyTrans) | OCR/LLM/erase-based translation + edit |
| B4 | End-to-end IIMT | **Translatotron-V** | [arxiv:2407.02894](https://arxiv.org/html/2407.02894v1) | Structure-aware end-to-end baseline |
| B5 | Practical IIMT | **VisTrans / PRIM** | [huggingface.co/yztian/VisTrans](https://huggingface.co/yztian/VisTrans) | Multilingual layout-preserving rewriting |
| B6 | MLLM analysis | **InImageTrans** | [arxiv:2605.29476](https://arxiv.org/pdf/2605.29476.pdf) | Hallucination/omission analysis baseline |
| B7 | Document IIMT | **DIMT25 open systems** | [ICDAR DIMT25 Track 2](https://huggingface.co/liangyupu/DIMT2025.ICDAR.Track_2) | Complex document layout comparison |
| **Ours** | — | **ReDesign text pipeline** | this repo | Text parse → edit → re-render |

> **Exclusion rule:** No closed APIs (GPT-4V, Gemini, proprietary IIMT). Optional ablation with open MLLMs (LLaVA, Qwen-VL) only if weights and inference code are public.

---

## 3. Benchmarks

| ID | Benchmark | Scenarios | Primary use | Split policy |
|----|-----------|-----------|-------------|--------------|
| D1 | **VISTRA** | Natural images, scene text | OCR vs translation error decomposition | Official dev/test; report per-language |
| D2 | **IMTBench** | Document / web / scene / PPT | Main end-to-end IIMT evaluation | Official splits; multi-scenario breakdown |
| D3 | **OCRMT30K** | Legacy TIT | Continuity with prior work | Standard train/dev/test from paper |
| D4 | **PRIM / IIMT30k** | Practical multilingual | Real-world poster/product cases | Official test set |
| D5 | **DIMT25** | Complex document layouts | Layout-heavy stress test | ICDAR Track 2 official split |

### 3.1 Benchmark split table (recording template)

| benchmark | split | subset | #images | src_langs | tgt_langs | notes |
|-----------|-------|--------|---------|-----------|-----------|-------|
| VISTRA | test | all | TBD | en | zh, ja, ko, … | per-language tables |
| VISTRA | test | scene_only | TBD | en | * | optional slice |
| IMTBench | test | document | TBD | * | * | main table |
| IMTBench | test | web | TBD | * | * | |
| IMTBench | test | scene | TBD | * | * | |
| IMTBench | test | ppt | TBD | * | * | |
| OCRMT30K | test | all | TBD | en | zh | legacy comparison |
| PRIM | test | all | TBD | en | zh, ja, ko, … | VisTrans default langs |
| DIMT25 | test | track2 | TBD | * | * | document IIMT |

Fill `#images` after downloading each dataset. Keep a frozen `data/manifests/{benchmark}_{split}.jsonl` per run.

---

## 4. Metrics

### 4.1 Translation quality
- **COMET** (primary), **BLEU**, **chrF**
- Report at image-level and corpus-level

### 4.2 OCR fidelity (on generated image)
- **CER**, **WER**, **exact match** after re-OCR
- Re-OCR engine: PaddleOCR (fixed version) for all systems

### 4.3 Visual fidelity
- **SSIM**, **LPIPS** (background mask excluding text regions)
- **Background preservation score** (IMTBench-style, if script available)

### 4.4 Layout fidelity
- Bounding-box **IoU** (predicted vs reference text regions)
- **Text occupancy ratio**, **overflow rate**, **line-break validity**

### 4.5 Efficiency
- **Latency** (s/image, warm GPU)
- **Peak VRAM** (GB)
- **Parameter count** (M)

### 4.6 Structure-aware (Translatotron-V lineage)
- **Structure-BLEU** where position annotations exist

---

## 5. Experimental protocol

### 5.1 Main comparison

For each `(image, src_lang, tgt_lang)` triple:

1. Run all baselines with **identical input** and **fixed target language**.
2. Save:
   - `pred_text.txt` — predicted translation string
   - `pred_image.png` — edited image
   - `reocr.json` — PaddleOCR on output image
   - `meta.json` — timing, GPU mem, model version, git hash

### 5.2 Stress tests (held-out slices)

| Test | Condition | Purpose |
|------|-----------|---------|
| **A** OCR-noise robustness | Corrupt OCR input (substitution, deletion) | Cascade error propagation |
| **B** Layout overflow | Target language longer than source | Typography adaptation |
| **C** Script transfer | en→{ko, ja, ar, zh} | Multilingual generalization |
| **D** Real-world design | Posters, ads, packaging | Practical usability |
| **E** Re-OCR consistency | OCR on generated image vs target | Text integrity |

### 5.3 Error decomposition

Manual or rule-assisted labeling into:

| Error type | Description |
|------------|-------------|
| `ocr_missing` | Source text not detected |
| `ocr_substitution` | Wrong source reading |
| `translation_omission` | Content dropped |
| `translation_hallucination` | Content added |
| `render_misplacement` | Wrong position/size |
| `font_style_mismatch` | Readable but wrong style |
| `background_artifact` | Non-text region damaged |

Report **error rate per 100 text regions** and **dominant failure mode per scenario**.

---

## 6. Model execution commands

Directory convention:

```
experiments/iimt/
├── envs/           # one venv/conda per baseline family
├── data/manifests/
├── outputs/{run_id}/{baseline}/{benchmark}/{sample_id}/
└── scripts/run_all.sh
```

### 6.1 B1 — PaddleOCR + MarianMT (cascade)

```bash
# Setup
python -m venv envs/cascade_paddle
source envs/cascade_paddle/bin/activate
pip install paddleocr paddlepaddle-gpu transformers sacrebleu torch

# Single image
python scripts/baselines/cascade_paddle_marian.py \
  --image data/samples/poster_001.png \
  --src_lang en --tgt_lang ko \
  --marian_model Helsinki-NLP/opus-mt-en-ko \
  --out_dir outputs/run001/B1/VISTRA/poster_001

# Batch (manifest jsonl)
python scripts/baselines/cascade_paddle_marian.py \
  --manifest data/manifests/vistra_test.jsonl \
  --out_root outputs/run001/B1/VISTRA
```

### 6.2 B2 — Tesseract + mBART

```bash
source envs/cascade_tesseract/bin/activate
python scripts/baselines/cascade_tesseract_mbart.py \
  --image data/samples/poster_001.png \
  --src_lang en_XX --tgt_lang ko_KR \
  --mbart_model facebook/mbart-large-50-many-to-many-mmt \
  --out_dir outputs/run001/B2/IMTBench/doc_042
```

### 6.3 B3 — AnyTrans

```bash
git clone https://github.com/qzp2018/AnyTrans.git third_party/AnyTrans
cd third_party/AnyTrans && pip install -r requirements.txt

python run.py \
  --input ../../data/samples/poster_001.png \
  --source en --target ko \
  --output ../../outputs/run001/B3/VISTRA/poster_001/pred_image.png
```

> Adjust CLI to AnyTrans repo's actual entry point after clone.

### 6.4 B4 — Translatotron-V

```bash
# Follow official repo instructions (arxiv 2407.02894)
python scripts/baselines/translatotron_v_infer.py \
  --checkpoint checkpoints/translatotron_v.pt \
  --image data/samples/poster_001.png \
  --tgt_lang ko \
  --out_dir outputs/run001/B4/VISTRA/poster_001
```

### 6.5 B5 — VisTrans / PRIM

```bash
pip install transformers diffusers accelerate
python scripts/baselines/vistrans_infer.py \
  --model yztian/VisTrans \
  --image data/samples/poster_001.png \
  --src_lang en --tgt_lang zh \
  --out_dir outputs/run001/B5/PRIM/poster_001
```

### 6.6 B6 — InImageTrans

```bash
# Use paper-provided open implementation
python scripts/baselines/inimagetrans_infer.py \
  --image data/samples/poster_001.png \
  --tgt_lang ko \
  --out_dir outputs/run001/B6/IMTBench/scene_017
```

### 6.7 B7 — DIMT25 open systems

```bash
# Example: ICDAR Track 2 baseline from HuggingFace
python scripts/baselines/dimt25_infer.py \
  --model liangyupu/DIMT2025.ICDAR.Track_2 \
  --image data/samples/doc_001.png \
  --tgt_lang zh \
  --out_dir outputs/run001/B7/DIMT25/doc_001
```

### 6.8 Ours — ReDesign text pipeline

```bash
cd web/ideogram-editor
source ../../.venv-text/bin/activate
python server.py --host 127.0.0.1 --port 8780 &

# CLI batch (to implement)
python ../../scripts/run_text_extract.py \
  --image data/samples/poster_001.png \
  --output_dir outputs/run001/Ours/VISTRA/poster_001 \
  --tgt_lang ko
```

### 6.9 Evaluation (all baselines)

```bash
python scripts/eval/compute_metrics.py \
  --pred_root outputs/run001 \
  --ref_manifest data/manifests/vistra_test.jsonl \
  --out_csv results/run001_vistra_metrics.csv

python scripts/eval/error_decomposition.py \
  --pred_root outputs/run001 \
  --out_csv results/run001_error_breakdown.csv
```

---

## 7. Results CSV schema

### 7.1 `results/{run_id}_per_sample.csv`

| column | type | description |
|--------|------|-------------|
| `run_id` | str | e.g. `run001` |
| `baseline_id` | str | B1–B7, Ours |
| `baseline_name` | str | human-readable |
| `benchmark` | str | VISTRA, IMTBench, … |
| `split` | str | test / stress_A / … |
| `scenario` | str | document, scene, ppt, … |
| `sample_id` | str | unique within benchmark |
| `image_path` | str | input image |
| `src_lang` | str | ISO or paper convention |
| `tgt_lang` | str | |
| `ref_text` | str | ground-truth translation |
| `pred_text` | str | model output text |
| `pred_image_path` | str | |
| `reocr_text` | str | OCR on output |
| `bleu` | float | |
| `chrf` | float | |
| `comet` | float | |
| `cer` | float | ref vs reocr |
| `wer` | float | |
| `exact_match` | int | 0/1 |
| `ssim_bg` | float | background only |
| `lpips_bg` | float | |
| `bbox_iou` | float | |
| `overflow_rate` | float | 0–1 |
| `latency_sec` | float | |
| `peak_vram_gb` | float | |
| `dominant_error` | str | from taxonomy §5.3 |
| `model_version` | str | checkpoint / commit |
| `timestamp` | str | ISO8601 |

### 7.2 `results/{run_id}_aggregate.csv`

| column | type | description |
|--------|------|-------------|
| `run_id` | str | |
| `baseline_id` | str | |
| `benchmark` | str | |
| `scenario` | str | optional grouping |
| `tgt_lang` | str | optional grouping |
| `n_samples` | int | |
| `bleu_mean` | float | |
| `chrf_mean` | float | |
| `comet_mean` | float | |
| `cer_mean` | float | |
| `wer_mean` | float | |
| `exact_match_rate` | float | |
| `ssim_bg_mean` | float | |
| `lpips_bg_mean` | float | |
| `bbox_iou_mean` | float | |
| `overflow_rate_mean` | float | |
| `latency_mean_sec` | float | |
| `peak_vram_gb` | float | |
| `params_m` | float | |
| `notes` | str | |

### 7.3 `results/{run_id}_error_breakdown.csv`

| column | type | description |
|--------|------|-------------|
| `run_id` | str | |
| `baseline_id` | str | |
| `benchmark` | str | |
| `scenario` | str | |
| `error_type` | str | §5.3 taxonomy |
| `count` | int | |
| `rate_per_100_regions` | float | |

---

## 8. Run manifest (`data/manifests/*.jsonl`)

Each line:

```json
{
  "sample_id": "vistra_00042",
  "benchmark": "VISTRA",
  "split": "test",
  "scenario": "scene",
  "image_path": "data/VISTRA/images/00042.png",
  "src_lang": "en",
  "tgt_lang": "ko",
  "ref_text": "Sale ends Sunday",
  "ref_regions": [{"bbox": [10, 20, 200, 50], "text": "Sale ends Sunday"}]
}
```

---

## 9. Ablation plan

| Ablation | Variable | Levels |
|----------|----------|--------|
| OCR noise | Input corruption rate | 0%, 10%, 20% CER |
| Length expansion | tgt/src character ratio | short, medium, long |
| Script | tgt_lang | Latin, CJK, Arabic |
| Layout density | regions per image | sparse, medium, dense |
| Orientation | text angle | horizontal, rotated, curved |

---

## 10. Paper-ready paragraph (copy-paste)

**Objective.** We study open-source multilingual in-image rewriting: given an image containing source-language text, generate an edited image that preserves meaning, layout, typography, and background fidelity while translating the text into a target language.

**Baselines.** We evaluate only public and reproducible systems: OCR+MT cascades with PaddleOCR/Tesseract and MarianMT or mBART, AnyTrans, Translatotron-V, VisTrans/PRIM, InImageTrans, and open-source DIMT25 systems.

**Benchmarks.** We use VISTRA for visually-situated translation in natural images, IMTBench for multi-scenario end-to-end IIMT, OCRMT30K as a legacy TIT benchmark, and PRIM/DIMT25/IIMT30k-style datasets for practical document and layout-heavy scenarios.

**Metrics.** We report COMET, BLEU, chrF, CER, WER, SSIM, LPIPS, background preservation, layout overflow rate, and OCR re-read consistency. We additionally report latency, peak GPU memory, and parameter count.

**Ablations.** We analyze OCR noise, target-language length expansion, script type, layout density, and text orientation. We also separate errors into OCR miss, translation omission, hallucination, and rendering mismatch.

**Hypothesis.** Cascaded systems will fail mainly under OCR noise and layout overflow, while end-to-end systems will fail mainly under grounding and typography mismatch; practical multilingual systems should perform best on real-world poster/product cases.

---

## 11. Implementation checklist

- [ ] Download and freeze all benchmark manifests
- [ ] Clone and pin commits for each third-party baseline
- [ ] Implement unified `pred_image.png` + `pred_text.txt` output contract
- [ ] Implement `compute_metrics.py` with fixed re-OCR (PaddleOCR 3.x)
- [ ] Run smoke test: 5 images × all baselines
- [ ] Full test run: primary table (IMTBench + VISTRA)
- [ ] Stress tests A–E
- [ ] Error decomposition annotation (min 200 regions stratified)
- [ ] Generate LaTeX tables from aggregate CSV

---

## References

- VISTRA: [arxiv:2406.11432](https://arxiv.org/html/2406.11432v1)
- Translatotron-V: [arxiv:2407.02894](https://arxiv.org/html/2407.02894v1)
- IMTBench / InImageTrans: [arxiv:2605.29476](https://arxiv.org/pdf/2605.29476.pdf)
- DIMT25: [arxiv:2504.17315](https://arxiv.org/abs/2504.17315)
- VisTrans: [huggingface.co/yztian/VisTrans](https://huggingface.co/yztian/VisTrans)
- AnyTrans: [github.com/qzp2018/AnyTrans](https://github.com/qzp2018/AnyTrans)
