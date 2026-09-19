#!/usr/bin/env python3
"""E9 — score / launch news_tables sliced by (source, src_lang, tgt_lang)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths_util import sb_root

ROOT = sb_root()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=ROOT / "gold" / "news_eval.json")
    ap.add_argument("--dest-dir", type=Path, default=ROOT / "results" / "lang")
    ap.add_argument("--models", nargs="+", default=["qwen3vl"])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--limit", type=int, default=0, help="per language-pair cap (0=all)")
    ap.add_argument("--pairs", nargs="*", default=None, help="e.g. donga:en-ko chosun:en-ko")
    args = ap.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    by = defaultdict(list)
    for it in gold["items"]:
        key = f"{it.get('source')}:{it.get('src_lang')}-{it.get('tgt_lang')}"
        by[key].append(it)

    wanted = args.pairs or sorted(by.keys())
    args.dest_dir.mkdir(parents=True, exist_ok=True)
    meta = {"pairs": {}, "models": args.models}
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    for i, key in enumerate(wanted):
        items = by.get(key) or []
        if not items:
            print(f"skip empty {key}", flush=True)
            continue
        if args.limit:
            items = items[: args.limit]
        subset = {
            "task": "news_lang_pair",
            "pair": key,
            "n_items": len(items),
            "items": items,
        }
        gold_p = args.dest_dir / f"gold_{key.replace(':','_')}.json"
        gold_p.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
        dest = args.dest_dir / key.replace(":", "_")
        dest.mkdir(parents=True, exist_ok=True)
        gpu = gpus[i % len(gpus)]
        for m in args.models:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_news_tables.py"),
                "--models",
                m,
                "--gold",
                str(gold_p),
                "--dest-dir",
                str(dest),
                "--shard",
                "0/1",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["SELECTIVE_BENCH_ROOT"] = str(ROOT)
            print(f"[lang] GPU={gpu} pair={key} n={len(items)} model={m}", flush=True)
            subprocess.run(cmd, check=False, env=env)
            # score this pair
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "score_news_tables.py"), "--dest-dir", str(dest)],
                check=False,
            )
        meta["pairs"][key] = {"n": len(items), "dest": str(dest)}

    (args.dest_dir / "lang_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", args.dest_dir / "lang_meta.json")


if __name__ == "__main__":
    main()
