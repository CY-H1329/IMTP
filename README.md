# IMTP — Open-source IIMT + Selective News Experiment Harness

H100-ready harnesses for:

1. **IIMT baselines** — `experiments/iimt/` (AnyTrans / Translatotron-V / PRIM, …)
2. **Selective translation (news)** — `experiments/selective_news/` (KNOW / Decision / Preserve / guidedΔ / Method1)

## Quick start — selective news (4× GPU)

```bash
cd ~
git clone https://github.com/CY-H1329/IMTP.git
cd IMTP/experiments/selective_news
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip -r requirements.txt
source env.sh
export NEWS_DATA=$HOME/data/hf_news_pack   # rsync crops separately (~4.7GB)
export CUDA_VISIBLE_DEVICES=0,1,2,3
chmod +x run_4gpu.sh
./run_4gpu.sh smoke
```

Full protocol, experiment map (1–10), and paper-table scoring:

- [`experiments/selective_news/CONTEXT.md`](experiments/selective_news/CONTEXT.md)
- [`experiments/selective_news/H100_4GPU_RUN.md`](experiments/selective_news/H100_4GPU_RUN.md)

## Quick start — open-source IIMT

```bash
cd ~/IMTP/experiments/iimt
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip -r requirements.txt
source env.sh
export CUDA_VISIBLE_DEVICES=0
chmod +x setup.sh run_all.sh
./run_all.sh
```

## Layout

```
IMTP/
  README.md
  docs/EXPERIMENT_PROTOCOL_OPENSOURCE_IIMT.md
  experiments/
    samples/              # IIMT smoke PNGs
    iimt/                 # open-source IIMT harness
    selective_news/       # news selective-translation bench (this work)
      CONTEXT.md
      H100_4GPU_RUN.md
      run_4gpu.sh
      gold/               # news_eval + rules (GT frozen)
      scripts/
```

## Data note

`selective_news` gold JSON is in git. Image crops (`hf_news_pack`) are **not** — see `H100_4GPU_RUN.md` §2.

## Third-party (IIMT only, once)

```bash
mkdir -p ~/IMTP/experiments/third_party && cd ~/IMTP/experiments/third_party
git clone https://github.com/qzp2018/AnyTrans.git
git clone https://github.com/DeepLearnXMU/translatotron-v Translatotron-V
git clone https://github.com/BITHLP/PRIM
```

## HF / tokens

- `HF_TOKEN` for gated models (Gemma, PRIM, VisTrans, …)
- DIMT25: EULA + HF access (IIMT path)
