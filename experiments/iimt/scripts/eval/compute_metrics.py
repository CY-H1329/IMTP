#!/usr/bin/env python3
"""Compute per-sample and aggregate metrics from experiment outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common.manifest import load_manifest
from common.metrics_util import bbox_iou, cer, exact_match, run_paddle_ocr, wer

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None


def _bleu(ref: str, hyp: str) -> float:
    if not ref.strip():
        return 0.0
    try:
        import sacrebleu
        return float(sacrebleu.sentence_bleu(hyp, [ref]).score)
    except Exception:
        return 0.0


def _chrf(ref: str, hyp: str) -> float:
    if not ref.strip():
        return 0.0
    try:
        import sacrebleu
        return float(sacrebleu.sentence_chrf(hyp, [ref]).score)
    except Exception:
        return 0.0


def _ssim_bg(src: Path, pred: Path, mask_regions: List[Dict]) -> float:
    if ssim_fn is None or not src.exists() or not pred.exists():
        return float("nan")
    a = np.array(Image.open(src).convert("RGB"))
    b = np.array(Image.open(pred).convert("RGB"))
    if a.shape != b.shape:
        b_img = Image.fromarray(b).resize((a.shape[1], a.shape[0]))
        b = np.array(b_img)
    # Simple full-image SSIM as fallback; mask-aware version can be added later
    return float(ssim_fn(a, b, channel_axis=2, data_range=255))


def _mean_bbox_iou(ref_regions: List[Dict], pred_regions: List[Dict]) -> float:
    if not ref_regions:
        return float("nan")
    scores = []
    for ref in ref_regions:
        rb = ref.get("bbox")
        if not rb:
            continue
        best = 0.0
        for pred in pred_regions:
            pb = pred.get("bbox")
            if pb:
                best = max(best, bbox_iou(rb, pb))
        scores.append(best)
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_sample(
    record: Dict[str, Any],
    pred_dir: Path,
    src_image: Optional[Path],
) -> Dict[str, Any]:
    pred_text = (pred_dir / "pred_text.txt").read_text(encoding="utf-8") if (pred_dir / "pred_text.txt").exists() else ""
    pred_image = pred_dir / "pred_image.png"
    meta = json.loads((pred_dir / "meta.json").read_text(encoding="utf-8")) if (pred_dir / "meta.json").exists() else {}

    reocr_path = pred_dir / "reocr.json"
    if reocr_path.exists():
        reocr = json.loads(reocr_path.read_text(encoding="utf-8"))
    elif pred_image.exists():
        reocr = run_paddle_ocr(str(pred_image))
    else:
        reocr = {"full_text": "", "regions": []}

    ref_text = record.get("ref_text", "")
    reocr_text = reocr.get("full_text", "")

    row = {
        "run_id": record.get("run_id", ""),
        "baseline_id": meta.get("baseline_id", ""),
        "baseline_name": meta.get("baseline_name", ""),
        "benchmark": record.get("benchmark", ""),
        "split": record.get("split", ""),
        "scenario": record.get("scenario", ""),
        "sample_id": record.get("sample_id", pred_dir.name),
        "image_path": str(record.get("image_path", "")),
        "src_lang": record.get("src_lang", ""),
        "tgt_lang": record.get("tgt_lang", ""),
        "ref_text": ref_text,
        "pred_text": pred_text,
        "pred_image_path": str(pred_image) if pred_image.exists() else "",
        "reocr_text": reocr_text,
        "bleu": _bleu(ref_text, pred_text),
        "chrf": _chrf(ref_text, pred_text),
        "comet": float("nan"),
        "cer": cer(ref_text, reocr_text),
        "wer": wer(ref_text, reocr_text),
        "exact_match": exact_match(ref_text, reocr_text),
        "ssim_bg": _ssim_bg(src_image, pred_image, record.get("ref_regions", [])) if src_image else float("nan"),
        "lpips_bg": float("nan"),
        "bbox_iou": _mean_bbox_iou(record.get("ref_regions", []), reocr.get("regions", [])),
        "overflow_rate": float("nan"),
        "latency_sec": meta.get("latency_sec", float("nan")),
        "peak_vram_gb": float("nan"),
        "dominant_error": "",
        "model_version": meta.get("model_version", ""),
        "status": meta.get("status", ""),
        "timestamp": meta.get("timestamp", ""),
    }
    return row


def find_pred_dirs(pred_root: Path) -> List[Path]:
    if not pred_root.exists():
        return []
    return sorted(p for p in pred_root.rglob("meta.json") if p.parent.is_dir())


def main() -> None:
    p = argparse.ArgumentParser(description="Compute IIMT metrics")
    p.add_argument("--pred_root", type=Path, required=True, help="e.g. outputs/run001")
    p.add_argument("--ref_manifest", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--run_id", default="run001")
    args = p.parse_args()

    manifest = {r.sample_id: r for r in load_manifest(args.ref_manifest)}
    rows: List[Dict[str, Any]] = []

    for meta_path in find_pred_dirs(args.pred_root):
        pred_dir = meta_path.parent
        sid = pred_dir.name
        if sid not in manifest:
            continue
        rec = manifest[sid]
        record = {
            "run_id": args.run_id,
            "sample_id": sid,
            "benchmark": rec.benchmark,
            "split": rec.split,
            "scenario": rec.scenario,
            "image_path": str(rec.image_path),
            "src_lang": rec.src_lang,
            "tgt_lang": rec.tgt_lang,
            "ref_text": rec.ref_text,
            "ref_regions": rec.ref_regions,
        }
        rows.append(evaluate_sample(record, pred_dir, rec.image_path))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
