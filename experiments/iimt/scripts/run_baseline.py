#!/usr/bin/env python3
"""Unified baseline runner — dispatches by baseline ID."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.manifest import load_manifest

SCRIPT_MAP = {
    "B1": "baselines.cascade_paddle_marian",
    "B2": "baselines.cascade_tesseract_mbart",
    "B3": "baselines.anytrans_infer",
    "B4": "baselines.translatotron_v_infer",
    "B5": "baselines.vistrans_infer",
    "B6": "baselines.inimagetrans_infer",
    "B7": "baselines.dimt25_infer",
    "Ours": "baselines.redesign_infer",
}


def load_config() -> dict:
    cfg_path = ROOT / "config" / "baselines.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description="Run one IIMT baseline on a manifest")
    p.add_argument("--baseline", required=True, choices=list(SCRIPT_MAP.keys()))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--run_id", default="run001")
    p.add_argument("--marian_model")
    p.add_argument("--mbart_model")
    p.add_argument("--hf_model")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tool_gpus", default="0")
    args = p.parse_args()

    cfg = load_config()
    bl_cfg = (cfg.get("baselines") or {}).get(args.baseline, {})

    mod = importlib.import_module(SCRIPT_MAP[args.baseline])
    records = load_manifest(args.manifest)
    benchmark = records[0].benchmark if records else "custom"

    out_base = args.out_root / args.run_id / args.baseline / benchmark
    out_base.mkdir(parents=True, exist_ok=True)

    for rec in records:
        out_dir = out_base / rec.sample_id
        if args.baseline == "B1":
            pair = f"{rec.src_lang}-{rec.tgt_lang}"
            pair_cfg = (cfg.get("lang_pairs") or {}).get(pair, {})
            marian = (
                args.marian_model
                or pair_cfg.get("marian")
                or bl_cfg.get("default_marian", "Helsinki-NLP/opus-mt-tc-big-en-ko")
            )
            fallbacks = pair_cfg.get("marian_fallbacks") or [
                "Helsinki-NLP/opus-mt-tc-big-en-ko",
                "Helsinki-NLP/opus-mt-en-ko",
            ]
            mod.run_sample(rec, out_dir, marian, marian_fallbacks=fallbacks)
        elif args.baseline == "B2":
            mbart = args.mbart_model or bl_cfg.get("default_mbart", "facebook/mbart-large-50-many-to-many-mmt")
            mod.run_sample(rec, out_dir, mbart)
        elif args.baseline == "B5":
            model = args.hf_model or bl_cfg.get("hf_model", "yztian/VisTrans")
            mod.run_sample(rec, out_dir, model)
        elif args.baseline == "B7":
            model = args.hf_model or bl_cfg.get("hf_model", "liangyupu/DIMT2025.ICDAR.Track_2")
            mod.run_sample(rec, out_dir, model)
        elif args.baseline == "B4":
            mod.run_sample(rec, out_dir, args.checkpoint)
        elif args.baseline == "Ours":
            mod.run_sample(rec, out_dir, args.tool_gpus)
        else:
            mod.run_sample(rec, out_dir)

    print(f"Done: {args.baseline} → {out_base} ({len(records)} samples)")


if __name__ == "__main__":
    main()
