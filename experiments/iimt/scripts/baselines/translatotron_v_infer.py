#!/usr/bin/env python3
"""B4: Translatotron-V wrapper."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.io import save_outputs
from common.manifest import SampleRecord, load_manifest
from common.metrics_util import Timer, run_paddle_ocr

BASELINE_ID = "B4"
BASELINE_NAME = "Translatotron-V"
TP_DIR = REPO / "third_party" / "Translatotron-V"


def run_sample(record: SampleRecord, out_dir: Path, checkpoint: Path | None) -> Path:
    timer = Timer()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_image = out_dir / "pred_image.png"

    if not TP_DIR.exists():
        return save_outputs(
            out_dir,
            pred_text="",
            baseline_id=BASELINE_ID,
            baseline_name=BASELINE_NAME,
            record_meta={"sample_id": record.sample_id, "benchmark": record.benchmark},
            latency_sec=timer.elapsed(),
            status="setup_required",
            error=f"Install Translatotron-V under {TP_DIR}",
            extra={"paper": "https://arxiv.org/html/2407.02894v1"},
        )

    infer = TP_DIR / "infer.py"
    if not infer.exists():
        infer = next(TP_DIR.glob("**/infer*.py"), None)

    if infer is None:
        return save_outputs(
            out_dir,
            pred_text="",
            baseline_id=BASELINE_ID,
            baseline_name=BASELINE_NAME,
            record_meta={"sample_id": record.sample_id, "benchmark": record.benchmark},
            latency_sec=timer.elapsed(),
            status="setup_required",
            error="No infer script found in Translatotron-V repo",
        )

    cmd = [
        sys.executable, str(infer),
        "--image", str(record.image_path),
        "--tgt_lang", record.tgt_lang,
        "--output", str(pred_image),
    ]
    if checkpoint:
        cmd.extend(["--checkpoint", str(checkpoint)])

    proc = subprocess.run(cmd, cwd=TP_DIR, capture_output=True, text=True)
    pred_text = run_paddle_ocr(str(pred_image)).get("full_text", "") if pred_image.exists() else ""
    status = "ok" if proc.returncode == 0 and pred_image.exists() else "error"

    return save_outputs(
        out_dir,
        pred_text=pred_text,
        pred_image_path=pred_image if pred_image.exists() else None,
        reocr=run_paddle_ocr(str(pred_image)) if pred_image.exists() else None,
        baseline_id=BASELINE_ID,
        baseline_name=BASELINE_NAME,
        record_meta={"sample_id": record.sample_id, "benchmark": record.benchmark},
        latency_sec=timer.elapsed(),
        status=status,
        error=proc.stderr[:500] if status != "ok" else "",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=BASELINE_NAME)
    p.add_argument("--image", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--out_root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()

    if args.manifest:
        for rec in load_manifest(args.manifest):
            run_sample(rec, args.out_root / rec.sample_id, args.checkpoint)
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
    run_sample(rec, args.out_dir, args.checkpoint)


if __name__ == "__main__":
    main()
