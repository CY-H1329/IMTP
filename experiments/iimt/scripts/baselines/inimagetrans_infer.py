#!/usr/bin/env python3
"""B6: InImageTrans wrapper."""
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

BASELINE_ID = "B6"
BASELINE_NAME = "InImageTrans"
IIT_DIR = REPO / "third_party" / "InImageTrans"


def run_sample(record: SampleRecord, out_dir: Path) -> Path:
    timer = Timer()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_image = out_dir / "pred_image.png"

    if not IIT_DIR.exists():
        return save_outputs(
            out_dir,
            pred_text="",
            baseline_id=BASELINE_ID,
            baseline_name=BASELINE_NAME,
            record_meta={"sample_id": record.sample_id, "benchmark": record.benchmark},
            latency_sec=timer.elapsed(),
            status="setup_required",
            error=f"Clone InImageTrans to {IIT_DIR}",
            extra={"paper": "https://arxiv.org/pdf/2605.29476.pdf"},
        )

    infer = next(IIT_DIR.glob("**/*infer*.py"), None) or IIT_DIR / "run.py"
    if not infer.exists():
        return save_outputs(
            out_dir,
            pred_text="",
            baseline_id=BASELINE_ID,
            baseline_name=BASELINE_NAME,
            record_meta={"sample_id": record.sample_id, "benchmark": record.benchmark},
            latency_sec=timer.elapsed(),
            status="setup_required",
            error="No inference entry point in InImageTrans repo",
        )

    cmd = [
        sys.executable, str(infer),
        "--image", str(record.image_path),
        "--tgt_lang", record.tgt_lang,
        "--output", str(pred_image),
    ]
    proc = subprocess.run(cmd, cwd=IIT_DIR, capture_output=True, text=True)
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
    args = p.parse_args()

    if args.manifest:
        for rec in load_manifest(args.manifest):
            run_sample(rec, args.out_root / rec.sample_id)
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
    run_sample(rec, args.out_dir)


if __name__ == "__main__":
    main()
