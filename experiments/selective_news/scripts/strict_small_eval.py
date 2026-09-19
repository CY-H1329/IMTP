#!/usr/bin/env python3
"""Strict small-sample eval: Decision / Preserve / Gen / Guided.

Cold scoring:
  - Every gold span MUST appear in the model JSON (matched).
  - Missing a gold span = wrong for that span.
  - Item-level: if ANY gold span is missing or wrong → item fails (all-or-nothing).
  - Extra model spans are IGNORED (not penalized).
  - Gen/Guided scored on parsed decisions (+ preserve: output≈source).

Models: qwen3vl, internvl35. Default n=25 news articles.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths_util import sb_root  # noqa: E402

ROOT = sb_root()
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("SELECTIVE_BENCH_ROOT", str(ROOT))

from oracle_guide import (  # noqa: E402
    TGT_NAME,
    guided_prompt,
    inventory,
    rules_block,
    similar,
)
from run_news_tables import (  # noqa: E402
    decision_prompt,
    know_prompt,
    parse_json_items,
    resolve_image,
)
from run_rule_yesno import ALIASES, LocalVLM  # noqa: E402

PROTO = json.loads((ROOT / "gold" / "probe_chain_protocol.json").read_text())
GOLD = ROOT / "gold" / "news_eval.json"
DEST = ROOT / "results" / "strict_small"
BLANK = ROOT / "results" / "_blank.png"


def parse_loose(raw: str) -> list[dict]:
    items = parse_json_items(raw)
    if items:
        return items
    out = []
    for m in re.finditer(r"\{[^{}]+\}", raw or ""):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and (obj.get("text") or obj.get("decision")):
            out.append(obj)
    return out


def match_decision(inv: list[dict], preds: list[dict], check_output: bool = False) -> list[dict]:
    """Match each gold span. Missing → ok=False. Extra preds ignored."""
    rows = []
    for r in inv:
        pred = None
        out = None
        for p in preds:
            txt = p.get("text") or p.get("span") or ""
            if similar(txt, r["text"]) or similar(r["text"], txt):
                d = (p.get("decision") or "").lower()
                if d in ("translate", "preserve"):
                    pred = d
                    out = p.get("output")
                    break
        if pred is None:
            ok = False
            reason = "missing"
        elif pred != r["decision"]:
            ok = False
            reason = "wrong_decision"
        elif check_output and r["decision"] == "preserve":
            # preserve: output must keep source (or equal text)
            if out is None or out == "":
                ok = False
                reason = "preserve_no_output"
            elif similar(out, r["text"]) or similar(r["text"], str(out)):
                ok = True
                reason = "ok"
            else:
                ok = False
                reason = "preserve_rewritten"
        else:
            ok = True
            reason = "ok"
        rows.append(
            {
                "taxonomy": r["taxonomy"],
                "span": r["text"][:200],
                "expect": r["decision"],
                "pred": pred,
                "output": (str(out)[:120] if out is not None else None),
                "matched": pred is not None,
                "missing": pred is None,
                "ok": ok,
                "reason": reason,
            }
        )
    return rows


def span_acc(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(1 for r in rows if r.get("ok"))
    miss = sum(1 for r in rows if r.get("missing"))
    wrong = sum(1 for r in rows if r.get("matched") and not r.get("ok"))
    return {
        "n": n,
        "ok": ok,
        "acc": (ok / n) if n else None,
        "missing": miss,
        "missing_rate": (miss / n) if n else None,
        "wrong_decision": wrong,
    }


def item_perfect(rows: list[dict]) -> bool:
    """All-or-nothing: one miss or wrong → item fail."""
    return bool(rows) and all(r.get("ok") for r in rows)


def subset(rows, expect=None, tax=None):
    out = rows
    if expect:
        out = [r for r in out if r.get("expect") == expect]
    if tax:
        out = [r for r in out if r.get("taxonomy") == tax]
    return out


def pick_items(gold: dict, n: int) -> list[dict]:
    out = []
    for it in gold["items"]:
        inv = inventory(it)
        if not inv:
            continue
        taxes = {r["taxonomy"] for r in inv}
        if "PHOTO_SIGN" not in taxes:
            continue
        if not any(r["decision"] == "preserve" for r in inv):
            continue
        if not any(r["decision"] == "translate" for r in inv):
            continue
        if not resolve_image(it["image"], it).exists():
            continue
        out.append(it)
        if len(out) >= n:
            break
    return out


def load_done(path: Path) -> dict:
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done[rec["id"]] = rec
    return done


def pack_condition(item_rows: list[list[dict]]) -> dict:
    flat = [x for rows in item_rows for x in rows]
    perfect = sum(1 for rows in item_rows if item_perfect(rows))
    return {
        "span": span_acc(flat),
        "item_perfect": {
            "n": len(item_rows),
            "ok": perfect,
            "acc": (perfect / len(item_rows)) if item_rows else None,
        },
        "preserve_span": span_acc(subset(flat, expect="preserve")),
        "photo_sign_span": span_acc(subset(flat, tax="PHOTO_SIGN")),
        "translate_span": span_acc(subset(flat, expect="translate")),
    }


def unguided_prompt(item: dict) -> str:
    tgt_n = TGT_NAME.get(item.get("tgt_lang") or "ko", "Korean")
    return PROTO["prompts"]["S4_generate"].format(tgt_n=tgt_n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3vl"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--max-side", type=int, default=512)
    args = ap.parse_args()

    from PIL import Image

    args.dest.mkdir(parents=True, exist_ok=True)
    BLANK.parent.mkdir(parents=True, exist_ok=True)
    if not BLANK.exists():
        Image.new("RGB", (448, 448), "white").save(BLANK)

    gold = json.loads(GOLD.read_text())
    items = pick_items(gold, args.n)
    print(f"strict_small n={len(items)} dest={args.dest}", flush=True)

    for spec in args.models:
        mid = ALIASES.get(spec, spec)
        slug = mid.split("/")[-1].replace(" ", "_")
        jsonl = args.dest / f"{slug}.strict_n{args.n}.jsonl"
        done = load_done(jsonl)
        print(f"\n### {slug} resume {len(done)}/{len(items)}", flush=True)

        vlm = LocalVLM(mid, max_side=args.max_side, max_new=700)

        with jsonl.open("a", encoding="utf-8") as f:
            for i, it in enumerate(items):
                if it["id"] in done:
                    continue
                inv = inventory(it)
                img = resolve_image(it["image"], it)
                print(
                    f"=== [{i+1}/{len(items)}] {it['id']} spans={len(inv)} ===",
                    flush=True,
                )

                # Decision (spans listed + image)
                raw_d = vlm.generate(img, decision_prompt(it, inv), max_new=500)
                dec = match_decision(inv, parse_loose(raw_d or ""), check_output=False)

                # Gen unguided
                raw_u = vlm.generate(img, unguided_prompt(it), max_new=600)
                gen_u = match_decision(inv, parse_loose(raw_u or ""), check_output=True)

                # Guided
                raw_g = vlm.generate(img, guided_prompt(it, inv), max_new=600)
                gen_g = match_decision(inv, parse_loose(raw_g or ""), check_output=True)

                rec = {
                    "id": it["id"],
                    "image": str(img),
                    "src_lang": it.get("src_lang"),
                    "tgt_lang": it.get("tgt_lang"),
                    "n_gold": len(inv),
                    "decision": dec,
                    "gen_unguided": gen_u,
                    "gen_guided": gen_g,
                    "item_perfect": {
                        "decision": item_perfect(dec),
                        "preserve_decision": item_perfect(subset(dec, expect="preserve")),
                        "gen_unguided": item_perfect(gen_u),
                        "gen_guided": item_perfect(gen_g),
                    },
                    "raw": {
                        "decision": (raw_d or "")[:2500],
                        "unguided": (raw_u or "")[:2500],
                        "guided": (raw_g or "")[:2500],
                    },
                }

                def line(name, rows):
                    s = span_acc(rows)
                    p = item_perfect(rows)
                    pr = span_acc(subset(rows, expect="preserve"))
                    print(
                        f"  {name:12s} span={s['ok']}/{s['n']} "
                        f"miss={s['missing']} wrong={s['wrong_decision']} "
                        f"item_ok={int(p)}  preserve={pr['ok']}/{pr['n']}",
                        flush=True,
                    )

                line("Decision", dec)
                line("Gen", gen_u)
                line("Guided", gen_g)

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[it["id"]] = rec

        vlm.close()
        rows = [done[it["id"]] for it in items if it["id"] in done]

        summary = {
            "n_items": len(rows),
            "scoring": "strict: missing gold span = fail; extras ignored; item_perfect = all gold ok",
            "decision": pack_condition([r["decision"] for r in rows]),
            "gen_unguided": pack_condition([r["gen_unguided"] for r in rows]),
            "gen_guided": pack_condition([r["gen_guided"] for r in rows]),
        }
        # preserve-only item perfect from decision
        summary["preserve_decision"] = pack_condition(
            [subset(r["decision"], expect="preserve") for r in rows]
        )
        u = summary["gen_unguided"]["span"]["acc"]
        g = summary["gen_guided"]["span"]["acc"]
        summary["delta_span"] = (g - u) if (g is not None and u is not None) else None
        ui = summary["gen_unguided"]["item_perfect"]["acc"]
        gi = summary["gen_guided"]["item_perfect"]["acc"]
        summary["delta_item"] = (gi - ui) if (gi is not None and ui is not None) else None

        sp = args.dest / f"{slug}.summary.json"
        sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        def pct(x):
            return "—" if x is None else f"{100 * x:.1f}"

        lines = [
            f"# Strict small n={len(rows)} — {slug}",
            "",
            "Missing any gold span = wrong. Extras ignored. item_perfect = all gold spans correct.",
            "",
            "| Cond | span% | miss% | item_perfect% | preserve_span% | PHOTO_SIGN% |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, key in (
            ("Decision", "decision"),
            ("Preserve@Decision", "preserve_decision"),
            ("Gen unguided", "gen_unguided"),
            ("Guided", "gen_guided"),
        ):
            b = summary[key]
            lines.append(
                f"| {name} | {pct(b['span']['acc'])} | {pct(b['span']['missing_rate'])} | "
                f"{pct(b['item_perfect']['acc'])} | {pct(b['preserve_span']['acc'])} | "
                f"{pct(b['photo_sign_span']['acc'])} |"
            )
        lines += [
            "",
            f"- Δ Guided−Unguided (span) = {summary['delta_span']}",
            f"- Δ Guided−Unguided (item) = {summary['delta_item']}",
        ]
        md = args.dest / f"{slug}.summary.md"
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        print("wrote", jsonl, sp, md)


if __name__ == "__main__":
    main()
