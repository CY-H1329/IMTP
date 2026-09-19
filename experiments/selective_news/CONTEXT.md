# Background & context — Selective Translation (news) experiments
# For operators on a remote H100 box who pull https://github.com/CY-H1329/IMTP

## What this is

ICLR-oriented **selective image-to-text / visual localization** bench on **Dong-A + Chosun**
news crops. Claim under test:

> Open VLMs often **know** selective-translation rules verbally (KNOW-rules / KNOW spans)
> but **fail to bind** them when generating with the page image (Decision / Preserve / ACT / Generation).
> **Guided** prompts (oracle inventory of TRANSLATE vs PRESERVE spans, **no gold target strings**)
> measure how much of the gap is binding vs capability.

This tree lives at `experiments/selective_news/` inside IMTP (sibling of `experiments/iimt/`).

## Domain rules (frozen gold)

| Taxonomy | Decision | Meaning |
|---|---|---|
| HEADLINE | TRANSLATE | Editorial layer → follow official target edition |
| BODY | TRANSLATE | News body / standfirst |
| MASTHEAD | TRANSLATE | Wordmark localized per edition (Chosun/Dong-A practice in our gold) |
| PHOTO_SIGN | PRESERVE | Diegetic on-photo lettering (signs in the photo), not CMS chrome |

Gold spans + `gt_tgt` are **frozen**. Do not hand-edit masthead/logo labels.

## Eval set

- `gold/news_eval.json` — **2843** article items (Dong-A ~921, Chosun ~1922; mostly `en→ko`, few Dong-A `en→ja`)
- Images: prefer `NEWS_DATA=.../hf_news_pack` (`crops/<article_id>/<lang>.png`). Absolute paths inside gold are remapped automatically via `scripts/paths_util.py`.
- Pack build (on the machine that has the corpus): `python scripts/build_hf_news_pack.py` (paths inside that script may need editing once).

## Scoring (default = STRICT)

Missing any gold span in the model JSON = **fail** for that span. Extra spans ignored.
Soft blob `score_gen` is no longer used for Table 1/3. Outputs go to `results/news_tables_strict/`.

Progress logs look like: `[####------] 120/711 (16.9%)`.


| # | Name | What it measures | How to run |
|---|---|---|---|
| 1 | Rule KNOW | YES/NO on constructional rules, **blank** image | `./run_4gpu.sh e1` |
| 2 | KNOW spans | T/P on gold spans **without** page image | part of `tables` |
| 3 | Decision | T/P on gold spans **with** image | `tables` |
| 4 | Preserve | Accuracy on gold-P / PHOTO_SIGN | `tables` |
| 5 | GAP KNOW→ACT | Knew preserve at KNOW but rewrote in gen | scored in Table 1 |
| 6 | Translation | Gold-T span matches official target string | `tables` |
| 7 | Generation | Unguided S4 overall | `tables` |
| 8 | Guided vs unguided | Oracle inventory vs free gen (Δ) | Table 3 from `tables` |
| 9 | Language | Per `(source, src, tgt)` slice | `./run_4gpu.sh lang` |
| 10 | Method (mapping) | Bind-then-Act vs unguided vs oracle | `./run_4gpu.sh method` |

Protocol detail: each `run_news_tables` item = **4 isolated forwards** (KNOW, Decision, unguided S4, guided S4). No chat memory across probes.

## Method 1 (Bind-then-Act)

Inference-time only: model must emit a structured bind (span → translate/preserve) then act.
Compare to unguided and to oracle-guided. Ads `n=50` and news `n=50` / strict `n=25` scripts exist.
If Δ is small, treat as **analysis**, not the central paper contribution.

## Models (aliases)

| Alias | HF id | Notes |
|---|---|---|
| `qwen3vl` | Qwen/Qwen3-VL-8B-Instruct | default 8B |
| `internvl35` | OpenGVLab/InternVL3_5-8B-HF | |
| `gemma12` | google/gemma-3-12b-it | gated |
| `qwen38` | Qwen/Qwen3.8-27B | large; may need transformers git |
| `gemma27` | google/gemma-3-27b-it | gated, heavy |

## Outputs → paper tables

After runs:

```bash
./run_4gpu.sh score
# → results/paper_tables/INDEX.md
# → results/news_tables/table1.tex table3.tex tables.md
```

## Prior local findings (context only — re-run on H100 to confirm)

- Ads Method1 Decision@matched: roughly **+7pp** vs unguided (n=50).
- News strict small (n=25): large guided Δ on some metrics (~+50pp span-level) — scorer/JSON sensitivity matters; prefer **strict** miss=fail reporting.
- Full news tables: watch for negative/noisy Δ if free-form JSON parse fails; use strict_small + matched Decision as sanity.

## What is NOT in git

- `data/hf_news_pack` (~4.7GB crops) — copy/rsync separately or rebuild.
- HuggingFace weights — download on the H100.
- Large `results/` jsonl — gitignored; pull results back with rsync after runs.

## Related docs

- `H100_4GPU_RUN.md` — pull / venv / GPU 0–3 / every command
- `config/experiments.yaml` — machine-readable experiment map
- Parent IMTP `experiments/iimt/` — separate open-source IIMT harness (AnyTrans / Translatotron / PRIM); do not confuse with this selective_news bench.
