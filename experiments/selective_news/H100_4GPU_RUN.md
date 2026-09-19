# H100 ×4 — Pull, setup, run (GPUs 0,1,2,3)

Target machine: Linux + 4× NVIDIA H100 (or any 4 visible GPUs).  
Repo: https://github.com/CY-H1329/IMTP  
Harness: `experiments/selective_news/`  
Read `CONTEXT.md` first for science / table meaning.

---

## 0. One-time: clone

```bash
cd ~
git clone https://github.com/CY-H1329/IMTP.git
cd IMTP/experiments/selective_news
```

Later updates:

```bash
cd ~/IMTP && git pull
cd experiments/selective_news
```

---

## 1. Python env

```bash
cd ~/IMTP/experiments/selective_news
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# If Qwen3.8-27B fails to load:
# pip install -U git+https://github.com/huggingface/transformers.git
```

HuggingFace (gated Gemma / some Qwen):

```bash
huggingface-cli login
# or: export HF_TOKEN=hf_xxx
```

---

## 2. Data (`NEWS_DATA`)

Gold JSON is in git (`gold/news_eval.json`). **Crops are not.**

### Option A — copy the HF pack (~4.7GB)

From the machine that already has the pack:

```bash
# on source
rsync -avP /path/to/hf_news_pack/  h100:~/data/hf_news_pack/
```

On H100:

```bash
export NEWS_DATA=$HOME/data/hf_news_pack
# expected layout:
#   $NEWS_DATA/crops/<article_id>/{en,ko,ja,zh}.png
#   $NEWS_DATA/articles.jsonl
ls "$NEWS_DATA/crops" | head
```

### Option B — symlink into the experiment tree

```bash
mkdir -p ~/IMTP/experiments/selective_news/data
ln -sfn $HOME/data/hf_news_pack ~/IMTP/experiments/selective_news/data/hf_news_pack
```

`source env.sh` auto-detects `data/hf_news_pack` if `NEWS_DATA` is unset.

Image paths inside gold look like `/workspace/.../donga_corpus/.../en.png`.  
`scripts/paths_util.py` remaps them to `$NEWS_DATA/crops/donga_<id>/en.png` automatically.

---

## 3. Always source env + pin GPUs

```bash
cd ~/IMTP/experiments/selective_news
source .venv/bin/activate
source env.sh
export NEWS_DATA=${NEWS_DATA:-$HOME/data/hf_news_pack}
export CUDA_VISIBLE_DEVICES=0,1,2,3   # physical GPUs to use
export GPUS=0,1,2,3                   # indices *inside* the visible set
chmod +x run_4gpu.sh
nvidia-smi -L
```

`run_4gpu.sh` launches one process per GPU via `CUDA_VISIBLE_DEVICES=<one id>`.

---

## 4. Smoke test (mandatory before full jobs)

```bash
./run_4gpu.sh smoke
# → results/smoke/*.jsonl + tables.md
# log: logs/smoke.log
```

If images missing: fix `NEWS_DATA`. If OOM: lower side is already ~768 in tables / 512 in method.

---

## Scoring

**Decision** = finding alone (image + gold spans listed).  
**ACT** = unguided vs guided generation: Preserve `output≈source`, Translate `output≈gt_tgt` (no soft free pass).  
Results → `results/news_tables_act/`. Offline check: `python scripts/test_scoring_offline.py`.


| Goal | Command | GPUs | Wall-clock (rough) |
|---|---|---|---|
| E1 rules | `./run_4gpu.sh e1` | parallel models | minutes |
| E2–E8 tables | `./run_4gpu.sh tables` | 4 shards | many hours (2843×4 fwds) |
| E10 method | `./run_4gpu.sh method` | 2 models default | ~1–3h |
| Strict n=25 | `./run_4gpu.sh strict` | 2 models | ~30–90m |
| E9 language | `./run_4gpu.sh lang` | sequential per pair | hours |
| Score only | `./run_4gpu.sh score` | CPU | minutes |
| Everything | `./run_4gpu.sh all` | — | day-scale |

### Useful knobs

```bash
# Cap articles while debugging tables
LIMIT=32 ./run_4gpu.sh tables

# Different models on the 4 shards (one model per GPU)
MODELS="qwen3vl internvl35 gemma12 qwen3vl" ./run_4gpu.sh tables

# Same large model sharded 4 ways (default)
MODELS="qwen3vl qwen3vl qwen3vl qwen3vl" ./run_4gpu.sh tables

# Method / strict size
N_ART_METHOD=50 ./run_4gpu.sh method
N_STRICT=25 ./run_4gpu.sh strict

# Also run ads Method1 after news method
RUN_ADS_METHOD=1 ./run_4gpu.sh method
```

### Multi-model full sweep (recommended pattern)

```bash
# 8B pair first
MODELS="qwen3vl qwen3vl qwen3vl qwen3vl" ./run_4gpu.sh tables
# then InternVL (new dest still news_tables/; different slug files)
MODELS="internvl35 internvl35 internvl35 internvl35" ./run_4gpu.sh tables
# Gemma-12B
MODELS="gemma12 gemma12 gemma12 gemma12" ./run_4gpu.sh tables
./run_4gpu.sh score
```

### Detach from SSH (survives hangup)

```bash
setsid bash -lc 'source .venv/bin/activate; source env.sh; export NEWS_DATA=$HOME/data/hf_news_pack; ./run_4gpu.sh tables' \
  </dev/null >logs/tables_setsid.log 2>&1 &
echo $!
# monitor
tail -f logs/tables_qwen3vl_s0of4.log
nvidia-smi
```

Do **not** rely on bare `nohup ... &` inside notebooks — prefer `setsid` as above.

---

## 6. Paper tables

```bash
./run_4gpu.sh score
ls results/paper_tables/
# INDEX.md
# paper_table1.tex  paper_table3.tex  paper_tables.md
# table_e1_know_rules.md/.tex
# table_e9_lang.md
# table_e10_method.md
# experiment_column_map.md
```

Copy `*.tex` / markdown into the paper draft. Column ↔ experiment map is in `experiment_column_map.md` and `config/experiments.yaml`.

---

## 7. Manual one-liners (single GPU)

```bash
source env.sh
export NEWS_DATA=$HOME/data/hf_news_pack

CUDA_VISIBLE_DEVICES=0 python scripts/run_rule_yesno.py --models qwen3vl --out results/rule_yesno/qwen3vl.json

CUDA_VISIBLE_DEVICES=1 python scripts/run_news_tables.py \
  --models qwen3vl --gold gold/news_eval.json \
  --dest-dir results/news_tables --shard 0/4

CUDA_VISIBLE_DEVICES=2 python scripts/smoke_bind_method1.py --models qwen3vl --n 50 --dest results/bind_method

CUDA_VISIBLE_DEVICES=3 python scripts/strict_small_eval.py --models internvl35 --n 25 --dest results/strict_small

python scripts/score_news_tables.py --dest-dir results/news_tables
python scripts/score_all_tables.py --results-dir results --which all
```

---

## 8. Fetch results back to laptop

```bash
rsync -avP h100:~/IMTP/experiments/selective_news/results/ ./results_h100/
rsync -avP h100:~/IMTP/experiments/selective_news/logs/ ./logs_h100/
```

---

## 9. Layout reference

```
IMTP/
  README.md
  experiments/
    selective_news/
      CONTEXT.md              # science + experiment map
      H100_4GPU_RUN.md        # this file
      env.sh
      run_4gpu.sh
      requirements.txt
      config/{models,experiments}.yaml
      gold/                   # news_eval, rules, prompts, sample_500
      scripts/                # runners + scorers
      data/hf_news_pack →     # YOU create (not in git)
      results/                # jsonl + paper_tables/
      logs/
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `NEWS_DATA/crops missing` | rsync pack; check `ls $NEWS_DATA/crops` |
| Image path not found | `NEWS_DATA` wrong; or article_id folder missing in pack |
| CUDA OOM | use `--max-side 512` (method/strict); one model per GPU; close zombies (`nvidia-smi`, kill PID) |
| HF 401 / gated | `huggingface-cli login`; accept model license |
| Qwen3.8 import errors | install transformers from git; reserve ~80GB+ |
| Empty / weird Δ | re-score; inspect raw JSON in jsonl; run `strict` for miss=fail |
| Job dies after SSH logout | use `setsid` recipe in §5 |
| Resume | runners skip ids already in jsonl (`load_done`) — just re-launch same command |

---

## 11. Quick checklist

- [ ] `git clone` / `git pull`
- [ ] venv + `pip install -r requirements.txt`
- [ ] `NEWS_DATA` crops present
- [ ] `HF_TOKEN` if needed
- [ ] `./run_4gpu.sh smoke` OK
- [ ] `e1` → `tables` → `method` → `strict` → `lang` → `score`
- [ ] rsync `results/paper_tables` home
