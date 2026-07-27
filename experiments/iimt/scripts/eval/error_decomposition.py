#!/usr/bin/env python3
"""Rule-assisted error decomposition for IIMT outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from common.manifest import load_manifest
from common.metrics_util import cer, wer

ERROR_TYPES = [
    "ocr_missing",
    "ocr_substitution",
    "translation_omission",
    "translation_hallucination",
    "render_misplacement",
    "font_style_mismatch",
    "background_artifact",
    "ok",
]


def classify(ref: str, pred: str, reocr: str, status: str) -> str:
    if status not in ("ok", ""):
        return "render_misplacement"
    ref, pred, reocr = (ref or "").strip(), (pred or "").strip(), (reocr or "").strip()
    if not ref and reocr:
        return "translation_hallucination"
    if ref and not reocr:
        return "ocr_missing"
    if ref and pred and len(pred) < 0.6 * len(ref):
        return "translation_omission"
    if pred and len(pred) > 1.4 * max(len(ref), 1):
        return "translation_hallucination"
    if cer(ref, reocr) > 0.35:
        return "ocr_substitution"
    if wer(ref, reocr) > 0.25:
        return "translation_omission"
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser(description="Error decomposition")
    p.add_argument("--pred_root", type=Path, required=True)
    p.add_argument("--ref_manifest", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--run_id", default="run001")
    args = p.parse_args()

    manifest = {r.sample_id: r for r in load_manifest(args.ref_manifest)}
    counts: Dict[str, Counter] = defaultdict(Counter)
    per_sample: List[Dict] = []

    for meta_path in sorted(args.pred_root.rglob("meta.json")):
        pred_dir = meta_path.parent
        sid = pred_dir.name
        if sid not in manifest:
            continue
        rec = manifest[sid]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pred_text = (pred_dir / "pred_text.txt").read_text(encoding="utf-8") if (pred_dir / "pred_text.txt").exists() else ""
        reocr = json.loads((pred_dir / "reocr.json").read_text(encoding="utf-8")) if (pred_dir / "reocr.json").exists() else {}
        reocr_text = reocr.get("full_text", "")
        err = classify(rec.ref_text, pred_text, reocr_text, meta.get("status", ""))
        bid = meta.get("baseline_id", "unknown")
        counts[(bid, rec.benchmark, rec.scenario)][err] += 1
        per_sample.append({
            "run_id": args.run_id,
            "baseline_id": bid,
            "sample_id": sid,
            "benchmark": rec.benchmark,
            "scenario": rec.scenario,
            "error_type": err,
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_rows = []
    for (bid, bench, scen), ctr in counts.items():
        total = sum(ctr.values())
        for err, c in ctr.items():
            agg_rows.append({
                "run_id": args.run_id,
                "baseline_id": bid,
                "benchmark": bench,
                "scenario": scen,
                "error_type": err,
                "count": c,
                "rate_per_100_regions": round(100.0 * c / max(total, 1), 2),
            })

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()) if agg_rows else ERROR_TYPES)
        w.writeheader()
        w.writerows(agg_rows)

    per_path = args.out_csv.with_name(args.out_csv.stem + "_per_sample.csv")
    if per_sample:
        with per_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
            w.writeheader()
            w.writerows(per_sample)

    print(f"Wrote {len(agg_rows)} aggregate rows → {args.out_csv}")


if __name__ == "__main__":
    main()
