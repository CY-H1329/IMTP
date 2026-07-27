#!/usr/bin/env python3
"""B2: Tesseract + mBART cascade baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common.io import save_outputs
from common.manifest import SampleRecord, load_manifest
from common.metrics_util import Timer, run_paddle_ocr
from common.render import regions_to_pred_text, render_translations

BASELINE_ID = "B2"
BASELINE_NAME = "Tesseract + mBART"

MBART_LANG = {
    "en": "en_XX", "ko": "ko_KR", "zh": "zh_CN", "ja": "ja_XX",
    "fr": "fr_XX", "de": "de_DE", "es": "es_XX",
}


def _tesseract_available() -> bool:
    import shutil
    return shutil.which("tesseract") is not None


def _ocr_regions(image_path: Path) -> list[dict]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(f"pytesseract required: {e}") from e

    if not _tesseract_available():
        raise RuntimeError(
            "tesseract binary not found in PATH. "
            "Install without sudo: conda install -y -c conda-forge tesseract"
        )

    data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
    regions = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        regions.append({
            "bbox": [x, y, x + w, y + h],
            "text": txt,
            "confidence": float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0,
        })
    return regions


def _translate(text: str, model_name: str, src_lang: str, tgt_lang: str) -> str:
    if not text.strip():
        return ""
    import os
    import torch
    from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

    token = os.environ.get("IIMT_HF_TOKEN") or os.environ.get("HF_TOKEN")
    kwargs = {"token": token} if token else {"token": False}
    tok = MBart50TokenizerFast.from_pretrained(model_name, **kwargs)
    model = MBartForConditionalGeneration.from_pretrained(model_name, **kwargs)
    src = MBART_LANG.get(src_lang, src_lang)
    tgt = MBART_LANG.get(tgt_lang, tgt_lang)
    tok.src_lang = src
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=tok.lang_code_to_id[tgt], max_length=512)
    return tok.decode(out[0], skip_special_tokens=True)


def run_sample(
    record: SampleRecord,
    out_dir: Path,
    mbart_model: str,
) -> Path:
    timer = Timer()
    if not _tesseract_available():
        return save_outputs(
            out_dir,
            pred_text="",
            pred_image_path=None,
            reocr={"status": "skipped"},
            baseline_id=BASELINE_ID,
            baseline_name=BASELINE_NAME,
            record_meta={
                "sample_id": record.sample_id,
                "benchmark": record.benchmark,
                "src_lang": record.src_lang,
                "tgt_lang": record.tgt_lang,
            },
            latency_sec=timer.elapsed(),
            model_version=mbart_model,
            status="setup_required",
            error=(
                "tesseract binary missing. "
                "Run: conda install -y -c conda-forge tesseract "
                "(do NOT use sudo apt on shared servers)"
            ),
        )

    regions = _ocr_regions(record.image_path)
    translated = []
    for reg in regions:
        tgt = _translate(reg["text"], mbart_model, record.src_lang, record.tgt_lang)
        translated.append({**reg, "translated": tgt})

    pred_text = regions_to_pred_text(translated)
    pred_image = out_dir / "_tmp_pred.png"
    render_translations(record.image_path, translated, pred_image)
    reocr = run_paddle_ocr(str(pred_image))

    return save_outputs(
        out_dir,
        pred_text=pred_text,
        pred_image_path=pred_image,
        reocr=reocr,
        baseline_id=BASELINE_ID,
        baseline_name=BASELINE_NAME,
        record_meta={
            "sample_id": record.sample_id,
            "benchmark": record.benchmark,
            "src_lang": record.src_lang,
            "tgt_lang": record.tgt_lang,
        },
        latency_sec=timer.elapsed(),
        model_version=mbart_model,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=BASELINE_NAME)
    p.add_argument("--image", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--out_root", type=Path)
    p.add_argument("--mbart_model", default="facebook/mbart-large-50-many-to-many-mmt")
    args = p.parse_args()

    if args.manifest:
        for rec in load_manifest(args.manifest):
            run_sample(rec, args.out_root / rec.sample_id, args.mbart_model)
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
    run_sample(rec, args.out_dir, args.mbart_model)


if __name__ == "__main__":
    main()
