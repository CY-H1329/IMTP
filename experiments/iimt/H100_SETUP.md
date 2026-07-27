# H100 JupyterLab 배포 가이드

ReDesign IIMT 실험을 **H100 JupyterLab 서버**에서 pull 후 실행하는 방법입니다.

## 1. GitHub에서 받기

```bash
cd ~
git clone https://github.com/CY-H1329/IMTP.git
cd IMTP/experiments/iimt
```

## 2. 환경 설정 (격리)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip -r requirements.txt
source env.sh
export CUDA_VISIBLE_DEVICES=0
```

## 3. Smoke test

```bash
chmod +x setup.sh run_all.sh
./run_all.sh
```

## 4. Third-party (1회)

```bash
cd ../../third_party
git clone https://github.com/qzp2018/AnyTrans.git
git clone https://github.com/DeepLearnXMU/translatotron-v Translatotron-V
git clone https://github.com/BITHLP/PRIM
```

## 5. H100 ETA (단일 H100, IMTBench 2,500)

| Baseline | ETA |
|---|---|
| B1 Cascade | ~30–60분 |
| B5 VisTrans | ~20–45분 |
| 8 baselines 전체 | ~6–12시간 |
| smoke 3장 | ~5–15분 |

## 6. HF / 데이터 (별도 준비)

- `HF_TOKEN`: gated models (PRIM, VisTrans)
- IMTBench/VISTRA/PRIM: manifest + images 별도 다운로드
- DIMT25: EULA + HF access

상세: `RUNBOOK.md`, `PUSH_AND_DEPLOY.md`
