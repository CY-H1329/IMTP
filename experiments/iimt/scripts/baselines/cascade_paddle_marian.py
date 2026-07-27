#!/usr/bin/env python3
"""B1: PaddleOCR + MarianMT cascade baseline."""
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

BASELINE_ID = "B1"
BASELINE_NAME = "PaddleOCR + MarianMT"

# Cache tokenizer/model so we don't reload per region.
_marian_cache: dict = {}


def _scrub_bad_hf_credentials() -> None:
    """Remove stale HF tokens that cause 401 even on public Helsinki-NLP models."""
    import os
    from pathlib import Path

    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_HUB_TOKEN",
    ):
        os.environ.pop(key, None)

    # Prefer anonymous downloads for public Marian checkpoints.
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    token_candidates = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        token_candidates.append(Path(hf_home) / "token")
    token_candidates.extend([
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ])
    for path in token_candidates:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass

    try:
        from huggingface_hub import logout
        logout()
    except Exception:
        pass


def _ocr_regions(image_path: Path) -> list[dict]:
    data = run_paddle_ocr(str(image_path))
    if data.get("status") != "ok":
        raise RuntimeError(data.get("error", "PaddleOCR failed"))
    return data["regions"]


def _get_marian(model_name: str):
    if model_name not in _marian_cache:
        import os
        from transformers import MarianMTModel, MarianTokenizer

        _scrub_bad_hf_credentials()

        # Local path override: IIMT_MARIAN_DIR=/path/to/opus-mt-en-ko
        local = os.environ.get("IIMT_MARIAN_DIR")
        load_id = local if local and Path(local).exists() else model_name

        try:
            tok = MarianTokenizer.from_pretrained(load_id, token=False)
            model = MarianMTModel.from_pretrained(load_id, token=False)
        except Exception as first_err:
            # Last resort: valid interactive login token (gated clusters)
            token = os.environ.get("IIMT_HF_TOKEN")  # explicit opt-in only
            if not token:
                raise RuntimeError(
                    "Failed to load MarianMT (HF 401/network). "
                    "Fix: unset HF_TOKEN; rm HF token files; or "
                    "`huggingface-cli login` with a valid token; or "
                    "set IIMT_MARIAN_DIR to a local model folder. "
                    f"Original error: {first_err}"
                ) from first_err
            tok = MarianTokenizer.from_pretrained(load_id, token=token)
            model = MarianMTModel.from_pretrained(load_id, token=token)
        model.eval()
        _marian_cache[model_name] = (tok, model)
    return _marian_cache[model_name]


def _translate(text: str, model_name: str) -> str:
    if not text.strip():
        return ""
    import torch

    tok, model = _get_marian(model_name)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=512)
    return tok.decode(out[0], skip_special_tokens=True)


def run_sample(record: SampleRecord, out_dir: Path, marian_model: str) -> Path:
    timer = Timer()
    regions = _ocr_regions(record.image_path)
    translated_regions = []
    for reg in regions:
        src = reg.get("text", "")
        tgt = _translate(src, marian_model)
        translated_regions.append({**reg, "translated": tgt})

    pred_text = regions_to_pred_text(translated_regions)
    pred_image = out_dir / "_tmp_pred.png"
    render_translations(record.image_path, translated_regions, pred_image)
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
        model_version=marian_model,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=BASELINE_NAME)
    p.add_argument("--image", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--out_root", type=Path)
    p.add_argument("--marian_model", default="Helsinki-NLP/opus-mt-en-ko")
    args = p.parse_args()

    if args.manifest:
        records = load_manifest(args.manifest)
        for rec in records:
            out = args.out_root / rec.sample_id
            run_sample(rec, out, args.marian_model)
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
    run_sample(rec, args.out_dir, args.marian_model)


if __name__ == "__main__":
    main()
