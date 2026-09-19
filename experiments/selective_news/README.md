# selective_news — Selective Translation harness (Dong-A / Chosun)

H100-ready runners for the ICLR selective-translation news experiments (KNOW / Decision / Preserve / GAP / Translation / Generation / guidedΔ / language / Method1).

- **Context:** [`CONTEXT.md`](CONTEXT.md)
- **Deploy & run (4 GPUs):** [`H100_4GPU_RUN.md`](H100_4GPU_RUN.md)

```bash
git clone https://github.com/CY-H1329/IMTP.git
cd IMTP/experiments/selective_news
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip -r requirements.txt
# place hf_news_pack crops, then:
source env.sh
export NEWS_DATA=$HOME/data/hf_news_pack
export CUDA_VISIBLE_DEVICES=0,1,2,3
./run_4gpu.sh smoke
./run_4gpu.sh tables    # E2–E8
./run_4gpu.sh score     # paper tex/md under results/paper_tables/
```
