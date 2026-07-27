#!/usr/bin/env python3
"""Generate stress-test manifests from a base manifest."""
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common.manifest import load_manifest


def corrupt_text(text: str, rate: float, rng: random.Random) -> str:
    if not text or rate <= 0:
        return text
    chars = list(text)
    n = max(1, int(len(chars) * rate))
    for _ in range(n):
        i = rng.randrange(len(chars))
        op = rng.choice(["sub", "del"])
        if op == "del":
            chars[i] = ""
        else:
            chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def main() -> None:
    p = argparse.ArgumentParser(description="Build stress-test manifests")
    p.add_argument("--base_manifest", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    records = load_manifest(args.base_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # A: OCR noise — store corrupted ref for cascade injection
    path_a = args.out_dir / "stress_A_ocr_noise.jsonl"
    with path_a.open("w", encoding="utf-8") as f:
        for rec in records:
            raw = {
                "sample_id": rec.sample_id,
                "benchmark": rec.benchmark,
                "split": "stress",
                "scenario": rec.scenario,
                "image_path": str(rec.image_path.relative_to(args.base_manifest.parent)) if not rec.image_path.is_absolute() else str(rec.image_path),
                "src_lang": rec.src_lang,
                "tgt_lang": rec.tgt_lang,
                "ref_text": rec.ref_text,
                "ref_regions": rec.ref_regions,
                "stress_tag": "A_ocr_noise",
                "corrupted_ref_text": corrupt_text(rec.ref_text, 0.15, rng),
            }
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")

    # B: layout overflow — long target placeholder
    path_b = args.out_dir / "stress_B_overflow.jsonl"
    with path_b.open("w", encoding="utf-8") as f:
        for rec in records:
            raw = {
                "sample_id": rec.sample_id,
                "benchmark": rec.benchmark,
                "split": "stress",
                "scenario": rec.scenario,
                "image_path": str(rec.image_path),
                "src_lang": rec.src_lang,
                "tgt_lang": rec.tgt_lang,
                "ref_text": rec.ref_text + " " + ("extended " * 8),
                "ref_regions": rec.ref_regions,
                "stress_tag": "B_overflow",
            }
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")

    # C: script transfer — multiple target langs
    langs = ["ko", "ja", "zh", "ar"]
    path_c = args.out_dir / "stress_C_script.jsonl"
    with path_c.open("w", encoding="utf-8") as f:
        for rec in records:
            for tgt in langs:
                raw = {
                    "sample_id": f"{rec.sample_id}_{tgt}",
                    "benchmark": rec.benchmark,
                    "split": "stress",
                    "scenario": rec.scenario,
                    "image_path": str(rec.image_path),
                    "src_lang": "en",
                    "tgt_lang": tgt,
                    "ref_text": rec.ref_text,
                    "ref_regions": rec.ref_regions,
                    "stress_tag": "C_script",
                }
                f.write(json.dumps(raw, ensure_ascii=False) + "\n")

    # D/E: pass-through tags on same images
    for tag, name in [("D_realworld", "stress_D_realworld.jsonl"), ("E_reocr", "stress_E_reocr.jsonl")]:
        path = args.out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                raw = {
                    "sample_id": rec.sample_id,
                    "benchmark": rec.benchmark,
                    "split": "stress",
                    "scenario": "poster" if tag.startswith("D") else rec.scenario,
                    "image_path": str(rec.image_path),
                    "src_lang": rec.src_lang,
                    "tgt_lang": rec.tgt_lang,
                    "ref_text": rec.ref_text,
                    "ref_regions": rec.ref_regions,
                    "stress_tag": tag,
                }
                f.write(json.dumps(raw, ensure_ascii=False) + "\n")

    print(f"Wrote stress manifests to {args.out_dir}")


if __name__ == "__main__":
    main()
