#!/usr/bin/env python3
"""Download PRIM from HuggingFace and build manifests.

Requires:
  export HF_TOKEN=hf_xxxx
  huggingface-cli login   # and accept dataset terms on HF page if gated

Usage:
  python scripts/data/prepare_prim.py \
    --data_root data/raw/prim \
    --out_manifest_dir data/manifests \
    --tgt_langs ko,zh,ja,de
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="yztian/PRIM")
    p.add_argument("--data_root", type=Path, default=Path("data/raw/prim"))
    p.add_argument("--out_manifest_dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--tgt_langs", default="ko,zh,ja")
    p.add_argument("--split", default="test")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN (PRIM is often gated)")
        print("  export HF_TOKEN=hf_xxxx")
        print("  Also accept terms: https://huggingface.co/datasets/yztian/PRIM")
        return 1

    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install datasets huggingface_hub")
        return 1

    args.data_root.mkdir(parents=True, exist_ok=True)
    args.out_manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prim] downloading {args.repo} → {args.data_root}")
    try:
        local_dir = snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(args.data_root),
            token=token,
        )
        print(f"[prim] snapshot at {local_dir}")
    except Exception as e:
        print(f"[prim] snapshot_download failed: {e}")
        print("Trying datasets.load_dataset ...")
        try:
            ds = load_dataset(args.repo, token=token)
            print(ds)
            # Save a lightweight index if structure unknown
            meta_path = args.data_root / "dataset_info.txt"
            meta_path.write_text(str(ds), encoding="utf-8")
        except Exception as e2:
            print(f"FAILED: {e2}")
            return 1

    # Heuristic: find images + json/csv annotations under data_root
    images = list(args.data_root.rglob("*.png")) + list(args.data_root.rglob("*.jpg"))
    print(f"[prim] found {len(images)} images under {args.data_root}")

    # If authors ship a jsonl/csv, prefer it
    jsonls = list(args.data_root.rglob("*.jsonl")) + list(args.data_root.rglob("*.json"))
    print(f"[prim] annotation-like files: {len(jsonls)}")
    for j in jsonls[:20]:
        print(f"  - {j.relative_to(args.data_root)}")

    # Generic manifest: one row per image × tgt_lang (ref_text empty until mapped)
    tgt_langs = [x.strip() for x in args.tgt_langs.split(",") if x.strip()]
    out = args.out_manifest_dir / "prim_test.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for img in sorted(images):
            for lang in tgt_langs:
                row = {
                    "sample_id": f"{img.stem}_{lang}",
                    "benchmark": "PRIM",
                    "split": args.split,
                    "scenario": "realworld",
                    "image_path": str(img.resolve()),
                    "src_lang": "en",
                    "tgt_lang": lang,
                    "ref_text": "",
                    "ref_regions": [],
                    "note": "ref_text may be empty; fill from PRIM annotations if available",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1

    print(f"[prim] wrote {out} ({n} rows)")
    print("Inspect annotations under data/raw/prim and refine prepare_prim.py if refs missing.")
    print("Next: BASELINES=B1,B5 RUN_ID=prim001 MANIFEST=data/manifests/prim_test.jsonl ./run_all.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
