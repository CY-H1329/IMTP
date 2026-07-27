#!/usr/bin/env python3
"""Aggregate per-sample CSV into summary table."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _nanmean(vals):
    xs = [v for v in vals if v == v]  # drop nan
    return mean(xs) if xs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    args = p.parse_args()

    rows = list(csv.DictReader(args.in_csv.open(encoding="utf-8")))
    groups = defaultdict(list)
    for r in rows:
        key = (r["run_id"], r["baseline_id"], r["benchmark"], r.get("scenario", ""), r.get("tgt_lang", ""))
        groups[key].append(r)

    out = []
    for (run_id, bid, bench, scen, tgt), items in sorted(groups.items()):
        out.append({
            "run_id": run_id,
            "baseline_id": bid,
            "benchmark": bench,
            "scenario": scen,
            "tgt_lang": tgt,
            "n_samples": len(items),
            "bleu_mean": round(_nanmean([float(x["bleu"]) for x in items]), 3),
            "chrf_mean": round(_nanmean([float(x["chrf"]) for x in items]), 3),
            "cer_mean": round(_nanmean([float(x["cer"]) for x in items]), 3),
            "wer_mean": round(_nanmean([float(x["wer"]) for x in items]), 3),
            "exact_match_rate": round(_nanmean([float(x["exact_match"]) for x in items]), 3),
            "ssim_bg_mean": round(_nanmean([float(x["ssim_bg"]) for x in items]), 3),
            "bbox_iou_mean": round(_nanmean([float(x["bbox_iou"]) for x in items]), 3),
            "latency_mean_sec": round(_nanmean([float(x["latency_sec"]) for x in items]), 3),
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out:
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    print(f"Wrote {len(out)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
