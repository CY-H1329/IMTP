#!/usr/bin/env python3
"""Download Marian model WITHOUT huggingface_hub auth (raw HTTPS).

Usage:
  python scripts/download_marian_anon.py \
    --repo Helsinki-NLP/opus-mt-en-ko \
    --out_dir .cache/models/opus-mt-en-ko

Then:
  export IIMT_MARIAN_DIR=$PWD/.cache/models/opus-mt-en-ko
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Minimal files needed by MarianTokenizer / MarianMTModel
FILES = [
    "config.json",
    "tokenizer_config.json",
    "source.spm",
    "target.spm",
    "vocab.json",
    "pytorch_model.bin",
    # newer layouts may use these instead:
    "model.safetensors",
    "tokenizer.json",
    "generation_config.json",
]


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url)
    # Force anonymous: never send Authorization
    req.add_header("User-Agent", "IMTP-marian-anon-downloader/1.0")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
        print(f"  OK  {dest.name} ({dest.stat().st_size} bytes)")
        return True
    except urllib.error.HTTPError as e:
        print(f"  SKIP {dest.name}: HTTP {e.code}")
        return False
    except Exception as e:
        print(f"  FAIL {dest.name}: {e}")
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="Helsinki-NLP/opus-mt-en-ko")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--endpoint", default="https://huggingface.co")
    args = p.parse_args()

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} → {out}")
    print("(no Authorization header)")

    ok = 0
    for name in FILES:
        url = f"{args.endpoint}/{args.repo}/resolve/main/{name}"
        if download(url, out / name):
            ok += 1

    # Must have config + tokenizer pieces + weights
    need_any_weight = (out / "pytorch_model.bin").exists() or (out / "model.safetensors").exists()
    need_tok = (out / "source.spm").exists() or (out / "tokenizer.json").exists()
    need_cfg = (out / "config.json").exists()

    meta = {"repo": args.repo, "files_ok": ok, "need_cfg": need_cfg, "need_tok": need_tok, "need_weight": need_any_weight}
    (out / "download_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if need_cfg and need_tok and need_any_weight:
        print("\nSUCCESS. Run:")
        print(f"  export IIMT_MARIAN_DIR={out}")
        return 0

    print("\nINCOMPLETE download. Anonymous HF access may be blocked on this cluster.")
    print("Use: huggingface-cli login   OR copy model folder from another machine.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
