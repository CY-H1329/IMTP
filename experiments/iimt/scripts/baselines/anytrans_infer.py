#!/usr/bin/env python3
"""B3: AnyTrans wrapper (requires third_party/AnyTrans)."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.io import save_outputs
from common.manifest import SampleRecord, load_manifest
from common.metrics_util import Timer, run_paddle_ocr

BASELINE_ID = "B3"
BASELINE_NAME = "AnyTrans"
ANYTRANS_DIR = REPO / "third_party" / "AnyTrans"


def _find_entry() -> Path | None:
    for name in ("run.py", "inference.py", "main.py", "demo.py"):
        p = ANYTRANS_DIR / name
        if p.exists():
            return p
    return None


def run_sample(record: SampleRecord, out_dir: Path) -> Path:
    timer = Timer()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_image = out_dir / "pred_image.png"

    entry = _find_entry()
    if not ANYTRANS_DIR.exists() or entry is None:
        return save_outputs(
            out_dir,
            pred_text="",
            pred_image_path=None,
            reocr={"status": "skipped"},
            baseline_id=BASELINE_ID,
            baseline_name=BASELINE_NAME,
            record_meta={"sample_id": record.sample_id, "benchmark": record.benchmark},
            latency_sec=timer.elapsed(),
            status="setup_required",
            error=f"Clone AnyTrans to {ANYTRANS_DIR}",
            extra={"setup": "git clone https://github.com/qzp2018/AnyTrans.git third_party/AnyTrans"},
        )

    cmd = [
        sys.executable, str(entry),
        "--input", str(record.image_path),
        "--source", record.src_lang,
        "--target", record.tgt_lang,
        "--output", str(pred_image),
    ]
    proc = subprocess.run(cmd, cwd=ANYTRANS_DIR, capture_output=True, text=True)
    pred_text = ""
    txt_path = out_dir / "pred_text.txt"
    if txt_path.exists():
        pred_text = txt_path.read_text(encoding="utf-8")
    elif pred_image.exists():
        pred_text = run_paddle_ocr(str(pred_image)).get("full_text", "")

    status = "ok" if proc.returncode == 0 and pred_image.exists() else "error"
    reocr = run_paddle_ocr(str(pred_image)) if pred_image.exists() else {"status": "skipped"}

    return save_outputs(
        out_dir,
        pred_text=pred_text,
        pred_image_path=pred_image if pred_image.exists() else None,
        reocr=reocr,
        baseline_id=BASELINE_ID,
        baseline_name=BASELINE_NAME,
        record_meta={"sample_id": record.sample_id, "benchmark": record.benchmark},
        latency_sec=timer.elapsed(),
        status=status,
        error=proc.stderr[:500] if status != "ok" else "",
        extra={"cmd": cmd, "stdout": proc.stdout[:500]},
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
