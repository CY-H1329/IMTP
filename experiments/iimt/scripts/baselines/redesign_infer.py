#!/usr/bin/env python3
"""Ours: ReDesign text-only pipeline wrapper."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(REPO))

from common.io import save_outputs
from common.manifest import SampleRecord, load_manifest
from common.metrics_util import Timer, run_paddle_ocr

BASELINE_ID = "Ours"
BASELINE_NAME = "ReDesign text pipeline"


def run_sample(record: SampleRecord, out_dir: Path, tool_gpus: str) -> Path:
    timer = Timer()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "redesign_out"
    work.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO / "ReDesign" / "run_text_official.py"),
        "--image", str(record.image_path),
        "--output_dir", str(work),
        "--tool_gpus", tool_gpus,
    ]

    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    pred_image = out_dir / "pred_image.png"
    pred_text = ""

    # Find best output image from ReDesign parse
    candidates = list(work.rglob("inpaint*.png")) + list(work.rglob("extracted.png"))
    if candidates:
        pred_image.write_bytes(candidates[0].read_bytes())

    parse_json = next(work.rglob("parse.json"), None)
    if parse_json and parse_json.exists():
        data = json.loads(parse_json.read_text(encoding="utf-8"))
        texts = []
        for layer in data.get("layers", []):
            if layer.get("type") == "text":
                texts.append(layer.get("text", ""))
        pred_text = " ".join(t for t in texts if t)

    if not pred_text and pred_image.exists():
        pred_text = run_paddle_ocr(str(pred_image)).get("full_text", "")

    status = "ok" if proc.returncode == 0 else "error"
    return save_outputs(
        out_dir,
        pred_text=pred_text,
        pred_image_path=pred_image if pred_image.exists() else None,
        reocr=run_paddle_ocr(str(pred_image)) if pred_image.exists() else {"status": "skipped"},
        baseline_id=BASELINE_ID,
        baseline_name=BASELINE_NAME,
        record_meta={
            "sample_id": record.sample_id,
            "benchmark": record.benchmark,
            "src_lang": record.src_lang,
            "tgt_lang": record.tgt_lang,
        },
        latency_sec=timer.elapsed(),
        status=status,
        error=proc.stderr[:800] if status != "ok" else "",
        extra={"cmd": cmd, "work_dir": str(work)},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=BASELINE_NAME)
    p.add_argument("--image", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--out_root", type=Path)
    p.add_argument("--tool_gpus", default="0")
    args = p.parse_args()

    if args.manifest:
        for rec in load_manifest(args.manifest):
            run_sample(rec, args.out_root / rec.sample_id, args.tool_gpus)
        return
    if not args.image or not args.out_dir:
        p.error("Provide --image + --out_dir or --manifest + --out_root")
    rec = SampleRecord(
        sample_id=args.out_dir.name,
        benchmark="custom",
        split="test",
        scenario="general",
        image_path=args.image,
        src_lang="en",
        tgt_lang="ko",
    )
    run_sample(rec, args.out_dir, args.tool_gpus)


if __name__ == "__main__":
    main()
