#!/usr/bin/env python3
"""Clone VISTRA and build full per-language manifests.

Usage:
  python scripts/data/prepare_vistra.py \
    --data_root data/raw/vistra \
    --out_manifest_dir data/manifests

Produces:
  data/manifests/vistra_en-{de,es,ru,zh}.jsonl
  data/manifests/vistra_all.jsonl   # all 4 langs (4x rows)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = "https://github.com/esalesky/vistra-benchmark.git"
LANGS = ("de", "es", "ru", "zh")


def clone_or_update(dest: Path) -> None:
    if (dest / ".git").exists() or (dest / "images").exists():
        print(f"[vistra] using existing: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[vistra] cloning {REPO} → {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO, str(dest)],
        check=True,
    )


def build_manifests(root: Path, out_dir: Path) -> dict[str, int]:
    ann_dir = root / "annotations"
    img_dir = root / "images"
    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"Expected {ann_dir} and {img_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {lang: (out_dir / f"vistra_en-{lang}.jsonl").open("w", encoding="utf-8") for lang in LANGS}
    all_path = out_dir / "vistra_all.jsonl"
    all_f = all_path.open("w", encoding="utf-8")
    counts = {lang: 0 for lang in LANGS}

    for ann_path in sorted(ann_dir.glob("*.json")):
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        image_file = data.get("image_file") or f"{ann_path.stem}.png"
        image_path = (img_dir / image_file).resolve()
        if not image_path.exists():
            print(f"[warn] missing image: {image_path}")
            continue

        src_segments = data.get("transcript") or []
        src_text = " ".join(src_segments) if isinstance(src_segments, list) else str(src_segments)
        translations = data.get("translation") or {}
        ctx = data.get("requires_image_context") or {}
        category = data.get("category") or "scene"

        for lang in LANGS:
            tgt_segs = translations.get(lang) or []
            tgt_text = " ".join(tgt_segs) if isinstance(tgt_segs, list) else str(tgt_segs)
            row = {
                "sample_id": f"{ann_path.stem}_en-{lang}",
                "benchmark": "VISTRA",
                "split": "test",
                "scenario": str(category),
                "image_path": str(image_path),
                "src_lang": "en",
                "tgt_lang": lang,
                "ref_text": tgt_text,
                "src_text": src_text,
                "ref_regions": [],
                "requires_image_context": ctx.get(lang),
                "annotation_path": str(ann_path.resolve()),
            }
            line = json.dumps(row, ensure_ascii=False) + "\n"
            writers[lang].write(line)
            all_f.write(line)
            counts[lang] += 1

    for f in writers.values():
        f.close()
    all_f.close()
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=Path("data/raw/vistra"))
    p.add_argument("--out_manifest_dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--skip_clone", action="store_true")
    args = p.parse_args()

    if not args.skip_clone:
        clone_or_update(args.data_root)
    counts = build_manifests(args.data_root, args.out_manifest_dir)
    total = sum(counts.values())
    print("[vistra] manifests written:")
    for lang, n in counts.items():
        print(f"  vistra_en-{lang}.jsonl : {n}")
    print(f"  vistra_all.jsonl       : {total}")
    print("\nNext:")
    print("  BASELINES=B1 RUN_ID=vistra_b1_de MANIFEST=data/manifests/vistra_en-de.jsonl ./run_all.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
