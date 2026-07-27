# GitHub Push & H100 배포

## Push 대상

```
ReDesign/experiments/iimt/     # 실험 harness
ReDesign/experiments/samples/  # smoke 이미지 3장
ReDesign/docs/EXPERIMENT_PROTOCOL_OPENSOURCE_IIMT.md
```

제외: `.venv/`, `outputs/`, `results/`, `.cache/`, `.env`

## Push (ReDesign repo)

```bash
cd /root/Desktop/workspace/CY/ReDesign
git add experiments/iimt experiments/samples docs/EXPERIMENT_PROTOCOL_OPENSOURCE_IIMT.md
git commit -m "Add open-source IIMT experiment harness for H100 deployment"

# CY repo (생성 후)
git remote add cy https://github.com/CY-H1329/ReDesign.git
git push -u cy main
```

## H100 JupyterLab pull

```bash
git clone https://github.com/CY-H1329/ReDesign.git
cd ReDesign/experiments/iimt
./setup.sh && source env.sh && ./run_all.sh
```

## H100에서 추가 필요

| 항목 | 필수 |
|---|---|
| GitHub repo URL | yes |
| HF_TOKEN (gated) | PRIM/VisTrans 시 |
| torch + CUDA | yes |
| paddlepaddle-gpu | B1 |
| third_party clones | B3–B5 |
| checkpoints | B4,B5,B7,Ours |
| full benchmarks | 별도 다운로드 |
