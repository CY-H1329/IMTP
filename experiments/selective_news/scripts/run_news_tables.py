#!/usr/bin/env python3
"""News Table 1 (KNOW / Decision / Preserve / ACT / Translation / Generation)
and Table 3 (unguided vs guided S4) on news_eval.json.

Each item: 4 isolated forwards
  1. KNOW  — text-only T/P on gold spans
  2. Decision — image T/P on gold spans
  3. ACT/Translation/Generation — unguided S4
  4. Guided S4 (oracle inventory, no gold target strings)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import os

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths_util import resolve_image as _resolve_image  # noqa: E402
from paths_util import sb_root  # noqa: E402

ROOT = sb_root()
ICLR = ROOT.parent
os.environ.setdefault("SELECTIVE_BENCH_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from oracle_guide import (  # noqa: E402
    TGT_NAME,
    guided_prompt,
    inventory,
    rules_block,
    score_gen,
    similar,
)
from run_probe_chain import load_done, parse_tp  # noqa: E402
from run_rule_yesno import ALIASES, LocalVLM, fill_prompt, parse_yn  # noqa: E402
from run_rule_yesno import all_rules  # noqa: E402

GOLD = ROOT / "gold" / "news_eval.json"
PROTO = json.loads((ROOT / "gold" / "probe_chain_protocol.json").read_text())
DEFAULT_DEST = ROOT / "results" / "news_tables"
BLANK = ROOT / "results" / "_blank.png"

ALIASES.update(
    {
        "gemma27": "google/gemma-3-27b-it",
        "gemma-3-27b": "google/gemma-3-27b-it",
        "gemma12": "google/gemma-3-12b-it",
        "gemma-3-12b": "google/gemma-3-12b-it",
        "qwen3vl8": "Qwen/Qwen3-VL-8B-Instruct",
        "internvl35-8b": "OpenGVLab/InternVL3_5-8B-HF",
        "qwen25-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen38": "Qwen/Qwen3.8-27B",
        "qwen3.8-27b": "Qwen/Qwen3.8-27B",
        "qwen38-27b": "Qwen/Qwen3.8-27B",
    }
)


def resolve_image(p: str, item: dict | None = None) -> Path:
    aid = (item or {}).get("article_id")
    lang = (item or {}).get("src_lang")
    return _resolve_image(p, article_id=aid, src_lang=lang)


def parse_json_items(raw: str) -> list[dict]:
    t = (raw or "").strip()
    if "```" in t:
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "items" in obj:
            obj = obj["items"]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            return []
    return []


def know_prompt(item: dict, inv: list[dict]) -> str:
    tgt_n = TGT_NAME.get(item.get("tgt_lang") or "ko", "Korean")
    lines = "\n".join(f'- [{r["taxonomy"]}] "{r["text"]}"' for r in inv)
    return (
        f"A {tgt_n} edition of a news article is being produced.\n"
        "For each span below, answer TRANSLATE or PRESERVE. No image.\n\n"
        f"RULES:\n{rules_block('news')}\n\n"
        f"SPANS:\n{lines}\n\n"
        "Output JSON list: "
        '{"text":"...","decision":"translate"|"preserve"} for every span. '
        "No extra commentary."
    )


def decision_prompt(item: dict, inv: list[dict]) -> str:
    tgt_n = TGT_NAME.get(item.get("tgt_lang") or "ko", "Korean")
    lines = "\n".join(f'- [{r["taxonomy"]}] "{r["text"]}"' for r in inv)
    return (
        f"Look at this news page. Target language: {tgt_n}.\n"
        "For each listed span that is visible, decide TRANSLATE or PRESERVE.\n\n"
        f"RULES:\n{rules_block('news')}\n\n"
        f"SPANS:\n{lines}\n\n"
        "Output JSON list: "
        '{"text":"...","decision":"translate"|"preserve"}. '
        "No extra commentary."
    )


def match_pred(inv: list[dict], preds: list[dict]) -> list[dict]:
    rows = []
    for r in inv:
        pred = None
        for p in preds:
            txt = p.get("text") or p.get("span") or ""
            if similar(txt, r["text"]) or similar(r["text"], txt):
                d = (p.get("decision") or "").lower()
                if d in ("translate", "preserve"):
                    pred = d
                    break
        rows.append(
            {
                "taxonomy": r["taxonomy"],
                "span": r["text"][:200],
                "expect": r["decision"],
                "pred": pred,
                "ok": pred == r["decision"] if pred else False,
            }
        )
    return rows


def score_block(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(1 for x in rows if x.get("ok"))
    return {"n": n, "ok": ok, "acc": (ok / n) if n else None}


def run_item(vlm: LocalVLM, item: dict) -> dict:
    inv = inventory(item)
    img = resolve_image(item["image"], item)
    tgt = item.get("tgt_lang") or "ko"
    tgt_n = TGT_NAME.get(tgt, tgt)

    raw_k = vlm.generate(BLANK, know_prompt(item, inv), max_new=400)
    know_rows = match_pred(inv, parse_json_items(raw_k))

    raw_d = vlm.generate(img, decision_prompt(item, inv), max_new=400)
    dec_rows = match_pred(inv, parse_json_items(raw_d))

    raw_u = vlm.generate(img, PROTO["prompts"]["S4_generate"].format(tgt_n=tgt_n), max_new=500)
    s4_u = score_gen(inv, raw_u or "")

    raw_g = vlm.generate(img, guided_prompt(item, inv), max_new=500)
    s4_g = score_gen(inv, raw_g or "")

    def subset(rows, tax=None, expect=None):
        out = rows
        if tax:
            out = [x for x in out if x.get("taxonomy") == tax]
        if expect:
            out = [x for x in out if x.get("expect") == expect]
        return out

    return {
        "id": item["id"],
        "article_id": item.get("article_id"),
        "category": "news",
        "source": item.get("source"),
        "image": str(img),
        "src_lang": item.get("src_lang"),
        "tgt_lang": tgt,
        "know": know_rows,
        "decision": dec_rows,
        "S4_unguided": s4_u,
        "S4_guided": s4_g,
        "preserve_decision": subset(dec_rows, expect="preserve"),
        "preserve_act_unguided": subset(s4_u, expect="preserve"),
        "translate_act_unguided": subset(s4_u, expect="translate"),
        "photo_sign_unguided": subset(s4_u, tax="PHOTO_SIGN"),
        "raw": {
            "know": (raw_k or "")[:2500],
            "decision": (raw_d or "")[:2500],
            "unguided": (raw_u or "")[:2500],
            "guided": (raw_g or "")[:2500],
        },
    }


def run_know_rules(vlm: LocalVLM) -> dict:
    recs = []
    for r in all_rules():
        if r.get("split") != "news":
            continue
        raw = vlm.generate(BLANK, fill_prompt(r), max_new=16)
        yn = parse_yn(raw)
        recs.append(
            {
                "id": r["id"],
                "expect": r["expect"],
                "pred": yn,
                "ok": yn == r["expect"],
                "raw": (raw or "")[:200],
            }
        )
    return {"n_ok": sum(x["ok"] for x in recs), "n": len(recs), "items": recs}


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3vl"])
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--dest-dir", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n of items")
    args = ap.parse_args()
    shard_i, shard_n = map(int, args.shard.split("/"))
    BLANK.parent.mkdir(parents=True, exist_ok=True)
    if not BLANK.exists():
        Image.new("RGB", (448, 448), "white").save(BLANK)
    gold = json.loads(args.gold.read_text())
    items = gold["items"]
    if args.limit:
        items = items[: args.limit]
    items = [it for i, it in enumerate(items) if i % shard_n == shard_i]
    dest: Path = args.dest_dir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"news_tables gold={args.gold} shard={args.shard} n={len(items)}", flush=True)
    for spec in args.models:
        mid = ALIASES.get(spec, spec)
        slug = mid.split("/")[-1].replace(" ", "_")
        outp = dest / f"{slug}.shard{shard_i}of{shard_n}.jsonl"
        rules_p = dest / f"{slug}.know_rules.json"
        done = load_done(outp)
        vlm = LocalVLM(mid, max_side=768, max_new=500)
        if not rules_p.exists():
            kr = run_know_rules(vlm)
            rules_p.write_text(json.dumps(kr, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{slug} KNOW-rules {kr['n_ok']}/{kr['n']}", flush=True)
        n = 0
        with outp.open("a", encoding="utf-8") as f:
            for it in items:
                if it["id"] in done:
                    continue
                rec = run_item(vlm, it)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                ku = score_block(rec["know"])
                du = score_block(rec["decision"])
                su = score_block(rec["S4_unguided"])
                sg = score_block(rec["S4_guided"])
                print(
                    f"{slug} {it['id'][:48]} know={ku['ok']}/{ku['n']} "
                    f"dec={du['ok']}/{du['n']} S4u={su['ok']}/{su['n']} "
                    f"S4g={sg['ok']}/{sg['n']}",
                    flush=True,
                )
        vlm.close()
        print("wrote", outp, "new", n)


if __name__ == "__main__":
    main()
