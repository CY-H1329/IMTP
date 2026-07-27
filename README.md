# IMTP — Open-source IIMT Experiment Harness

H100-ready harness for Image-to-Image Machine Translation (IIMT) baselines and evaluation.

## Quick start (H100 JupyterLab)

```bash
cd ~
git clone https://github.com/CY-H1329/IMTP.git
cd IMTP/experiments/iimt
```

## Environment (isolated)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip -r requirements.txt
source env.sh
export CUDA_VISIBLE_DEVICES=0
```

## Smoke test

```bash
chmod +x setup.sh run_all.sh
./run_all.sh
```

## Layout

```
IMTP/
  README.md
  docs/EXPERIMENT_PROTOCOL_OPENSOURCE_IIMT.md
  experiments/
    samples/          # smoke-test PNGs
    iimt/             # harness (scripts, config, manifests, runbooks)
```

Manifest paths use `../../../samples/` from `experiments/iimt/data/manifests/` — keep this layout.

## Docs

- `experiments/iimt/H100_SETUP.md` — H100 deploy guide
- `experiments/iimt/RUNBOOK.md` — full runbook
- `docs/EXPERIMENT_PROTOCOL_OPENSOURCE_IIMT.md` — open-source experiment protocol

## Third-party (once)

```bash
cd ../../third_party   # create sibling under experiments/ or adjust as needed
mkdir -p third_party && cd third_party
git clone https://github.com/qzp2018/AnyTrans.git
git clone https://github.com/DeepLearnXMU/translatotron-v Translatotron-V
git clone https://github.com/BITHLP/PRIM
```

## HF / data

- `HF_TOKEN` for gated models (PRIM, VisTrans)
- IMTBench/VISTRA/PRIM: manifests + images prepared separately
- DIMT25: EULA + HF access

See `experiments/iimt/RUNBOOK.md` and `PUSH_AND_DEPLOY.md`.
