#!/usr/bin/env python3
"""B7: DIMT25 ICDAR Track 2 open baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common.io import save_outputs
from common.manifest import SampleRecord, load_manifest
from common.metrics_util import Timer, run_paddle_ocr

BASELINE_ID = "B7"
BASELINE_NAME = "DIMT25 Track2"


def _infer_dimt25(image_path: Path, model_id: str, tgt_lang: str, out_path: Path) -> str:
    try:
        from transformers import AutoModel, AutoProcessor
        import torch
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(f"transformers/torch required: {e}") from e

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.cuda()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=tgt_lang, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "images") and outputs.images:
        outputs.images[0].save(out_path)
    elif hasattr(model, "translate"):
        img = model.translate(image, tgt_lang=tgt_lang)
        img.save(out_path)
    else:
        raise RuntimeError("DIMT25 model loaded but no known API — update dimt25_infer.py")

    return run_paddle_ocr(str(out_path)).get("full_text", "")


def run_sample(record: SampleRecord, out_dir: Path, model_id: str) -> Path:
    timer = Timer()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_image = out_dir / "pred_image.png"

    try:
        pred_text = _infer_dimt25(record.image_path, model_id, record.tgt_lang, pred_image)
        status = "ok"
        error = ""
    except Exception as e:
        pred_text = ""
        status = "error"
        error = str(e)

    return save_outputs(
        out_dir,
        pred_text=pred_text,
        pred_image_path=pred_image if pred_image.exists() else None,
        reocr=run_paddle_ocr(str(pred_image)) if pred_image.exists() else {"status": "skipped"},
        baseline_id=BASELINE_ID,
        baseline_name=BASELINE_NAME,
        record_meta={"sample_id": record.sample_id, "benchmark": record.benchmark},
        latency_sec=timer.elapsed(),
        model_version=model_id,
        status=status,
        error=error,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=BASELINE_NAME)
    p.add_argument("--image", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--out_root", type=Path)
    p.add_argument("--model", default="liangyupu/DIMT2025.ICDAR.Track_2")
    args = p.parse_args()

    if args.manifest:
        for rec in load_manifest(args.manifest):
            run_sample(rec, args.out_root / rec.sample_id, args.model)
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
        tgt_lang="zh",
    )
    run_sample(rec, args.out_dir, args.model)


if __name__ == "__main__":
    main()
