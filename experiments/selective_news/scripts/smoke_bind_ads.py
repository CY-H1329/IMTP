#!/usr/bin/env python3
"""Method 1 on ADS: unguided vs bind_act vs oracle.

Scoring policy (Decision-first):
  - Only gold spans are scored.
  - Extra model OCR/spans are IGNORED.
  - Missing gold spans are reported (coverage miss) — cannot judge Decision.
  - Primary metric = Decision accuracy among MATCHED gold spans.
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
    score_gen,
    similar,
)
from run_news_tables import parse_json_items, resolve_image  # noqa: E402
from run_rule_yesno import ALIASES, LocalVLM  # noqa: E402

PROTO = json.loads((ROOT / "gold" / "probe_chain_protocol.json").read_text())
SAMPLE = ROOT / "gold" / "sample_500.json"
DEST = ROOT / "results" / "bind_ads_n50"


def load_ads(n: int) -> list[dict]:
    d = json.loads(SAMPLE.read_text())
    items = d["items"]["ads"]
    out = []
    for it in items:
        inv = inventory(it)
        if not inv:
            continue
        if not resolve_image(it["image"], it).exists():
            continue
        # need at least one T and one P for selective claim
        ds = {r["decision"] for r in inv}
        if ds < {"translate", "preserve"}:
            continue
        out.append(it)
        if len(out) >= n:
            break
    return out


def unguided_prompt(item: dict) -> str:
    tgt_n = TGT_NAME.get(item.get("tgt_lang") or "en", "English")
    return PROTO["prompts"]["S4_generate"].format(tgt_n=tgt_n)


def bind_act_prompt(item: dict) -> str:
    tgt_n = TGT_NAME.get(item.get("tgt_lang") or "en", "English")
    return (
        f"Localize this product advertisement image into {tgt_n}.\n\n"
        "You MUST follow Bind-then-Act. Do not jump to the final string.\n\n"
        f"RULES (constructional):\n{rules_block('ads')}\n\n"
        "TYPE vocabulary:\n"
        "- brand          → brand / company name (proper name)\n"
        "- product        → product name on the pack\n"
        "- copy           → marketing headline / slogan / CTA / claims on the ad layout\n"
        "- pack_print     → text printed on the physical pack itself\n"
        "- ingredient     → INCI / ingredients / SPF / net weight / model code on pack\n"
        "- other\n\n"
        "VISUAL_CONTEXT vocabulary:\n"
        "- on_pack_surface   → printed on the physical product / pack\n"
        "- ad_overlay        → graphic text overlaid on the ad (not pack print)\n"
        "- logo_mark         → brand mark / logo area\n"
        "- uncertain\n\n"
        "For each visible text span, output ONE JSON object IN THIS ORDER:\n"
        "  text, type, visual_context, rule, decision, output\n"
        "decision MUST follow type+visual_context+rule "
        "(brand/product/pack_print/ingredient / on_pack_surface → preserve; "
        "copy / ad_overlay → translate).\n"
        "Return a JSON list only. Extra spans beyond the main brand/product/copy are OK.\n"
        "No markdown fences."
    )


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


def match_decisions(inv: list[dict], preds: list[dict]) -> list[dict]:
    """Score only gold spans. Extra preds ignored. Missing → matched=False."""
    rows = []
    for r in inv:
        pred = typ = vc = rule = out = None
        for p in preds:
            txt = p.get("text") or p.get("span") or ""
            if similar(txt, r["text"]) or similar(r["text"], txt):
                d = (p.get("decision") or "").lower()
                if d in ("translate", "preserve"):
                    pred = d
                    typ = p.get("type")
                    vc = p.get("visual_context")
                    rule = (p.get("rule") or "")[:160]
                    out = (p.get("output") or "")[:120]
                    break
        rows.append(
            {
                "taxonomy": r["taxonomy"],
                "span": r["text"][:200],
                "expect": r["decision"],
                "pred": pred,
                "matched": pred is not None,
                "ok": pred == r["decision"] if pred is not None else False,
                "missing": pred is None,
                "type": typ,
                "visual_context": vc,
                "rule": rule,
                "output": out,
            }
        )
    return rows


def acc(rows: list[dict], key: str = "ok") -> dict:
    n = len(rows)
    ok = sum(1 for r in rows if r.get(key))
    return {"n": n, "ok": ok, "acc": (ok / n) if n else None}


def subset(rows, tax=None, expect=None, matched_only=False):
    out = rows
    if matched_only:
        out = [r for r in out if r.get("matched")]
    if tax:
        out = [r for r in out if r.get("taxonomy") == tax]
    if expect:
        out = [r for r in out if r.get("expect") == expect]
    return out


def pack(rows_list: list[list[dict]]) -> dict:
    all_r = [x for rows in rows_list for x in rows]
    matched = [x for x in all_r if x.get("matched")]
    missing = [x for x in all_r if x.get("missing")]
    return {
        # primary: Decision among gold spans the model actually emitted
        "decision_matched": acc(matched),
        # secondary: treat missing as fail (strict)
        "decision_strict": acc(all_r),
        "missing": {
            "n": len(missing),
            "rate": (len(missing) / len(all_r)) if all_r else None,
            "by_taxonomy": {},
        },
        "preserve_matched": acc(subset(all_r, expect="preserve", matched_only=True)),
        "translate_matched": acc(subset(all_r, expect="translate", matched_only=True)),
        "coverage": {
            "matched": len(matched),
            "n_gold": len(all_r),
            "rate": (len(matched) / len(all_r)) if all_r else None,
        },
        "n_extra_ignored": None,  # filled per-condition below if available
    }


def fill_missing_by_tax(pack_d: dict, rows_list: list[list[dict]]) -> None:
    from collections import Counter

    c = Counter()
    for rows in rows_list:
        for x in rows:
            if x.get("missing"):
                c[x.get("taxonomy") or "?"] += 1
    pack_d["missing"]["by_taxonomy"] = dict(c)


def load_done(path: Path) -> dict:
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done[rec["id"]] = rec
    return done


def count_extras(inv: list[dict], preds: list[dict]) -> int:
    extra = 0
    for p in preds:
        txt = p.get("text") or ""
        if not txt:
            continue
        if not any(similar(txt, r["text"]) or similar(r["text"], txt) for r in inv):
            extra += 1
    return extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3vl"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()

    items = load_ads(args.n)
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"bind_ads n={len(items)} dest={args.dest} oracle={not args.skip_oracle}", flush=True)

    for spec in args.models:
        mid = ALIASES.get(spec, spec)
        slug = mid.split("/")[-1].replace(" ", "_")
        jsonl = args.dest / f"{slug}.bind_ads_n50.jsonl"
        done = load_done(jsonl)
        print(f"resume {len(done)}/{len(items)}", flush=True)

        vlm = LocalVLM(mid, max_side=args.max_side, max_new=900)

        with jsonl.open("a", encoding="utf-8") as f:
            for i, it in enumerate(items):
                if it["id"] in done:
                    continue
                inv = inventory(it)
                img = resolve_image(it["image"], it)
                print(
                    f"\n=== [{i+1}/{len(items)}] {slug} {it['id'][:48]} "
                    f"spans={len(inv)} {it.get('src_lang')}→{it.get('tgt_lang')} ===",
                    flush=True,
                )

                raw_u = vlm.generate(img, unguided_prompt(it), max_new=450)
                raw_b = vlm.generate(img, bind_act_prompt(it), max_new=750)
                preds_u = parse_loose(raw_u or "")
                preds_b = parse_loose(raw_b or "")

                rec = {
                    "id": it["id"],
                    "category": "ads",
                    "image": str(img),
                    "src_lang": it.get("src_lang"),
                    "tgt_lang": it.get("tgt_lang"),
                    "n_gold_spans": len(inv),
                    "n_pred_unguided": len(preds_u),
                    "n_pred_bind": len(preds_b),
                    "n_extra_unguided": count_extras(inv, preds_u),
                    "n_extra_bind": count_extras(inv, preds_b),
                    "dec_unguided": match_decisions(inv, preds_u),
                    "dec_bind": match_decisions(inv, preds_b),
                    "unguided_blob": score_gen(inv, raw_u or ""),
                    "bind_blob": score_gen(inv, raw_b or ""),
                    "bind_parsed": [
                        {
                            "text": (x.get("text") or "")[:100],
                            "type": x.get("type"),
                            "visual_context": x.get("visual_context"),
                            "decision": x.get("decision"),
                        }
                        for x in preds_b[:20]
                    ],
                    "raw": {
                        "unguided": (raw_u or "")[:3000],
                        "bind_act": (raw_b or "")[:4000],
                    },
                }
                if not args.skip_oracle:
                    raw_o = vlm.generate(img, guided_prompt(it, inv), max_new=450)
                    preds_o = parse_loose(raw_o or "")
                    rec["dec_oracle"] = match_decisions(inv, preds_o)
                    rec["oracle_blob"] = score_gen(inv, raw_o or "")
                    rec["n_pred_oracle"] = len(preds_o)
                    rec["n_extra_oracle"] = count_extras(inv, preds_o)
                    rec["raw"]["oracle"] = (raw_o or "")[:3000]

                def fmt(name, dec, n_extra):
                    m = subset(dec, matched_only=True)
                    miss = sum(1 for x in dec if x.get("missing"))
                    a = acc(m)
                    print(
                        f"  {name:10s} Decision@matched={a['ok']}/{a['n']} "
                        f"missing_gold={miss}/{len(dec)} extra_ignored={n_extra}",
                        flush=True,
                    )

                fmt("unguided", rec["dec_unguided"], rec["n_extra_unguided"])
                fmt("bind_act", rec["dec_bind"], rec["n_extra_bind"])
                if "dec_oracle" in rec:
                    fmt("oracle", rec["dec_oracle"], rec["n_extra_oracle"])

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[it["id"]] = rec

        vlm.close()
        rows = [done[it["id"]] for it in items if it["id"] in done]

        summary = {"n_items": len(rows), "category": "ads", "scoring": "Decision on matched gold only; extras ignored; missing reported"}
        summary["decision"] = {}
        for cond, key, extra_key in (
            ("unguided", "dec_unguided", "n_extra_unguided"),
            ("bind_act", "dec_bind", "n_extra_bind"),
            ("oracle", "dec_oracle", "n_extra_oracle"),
        ):
            if not rows or key not in rows[0]:
                continue
            p = pack([r[key] for r in rows])
            fill_missing_by_tax(p, [r[key] for r in rows])
            extras = [r.get(extra_key, 0) for r in rows]
            p["n_extra_ignored"] = {"sum": sum(extras), "mean": sum(extras) / len(extras) if extras else None}
            summary["decision"][cond] = p

        if "oracle" in summary["decision"] and "unguided" in summary["decision"]:
            u = summary["decision"]["unguided"]["decision_matched"]["acc"]
            b = summary["decision"]["bind_act"]["decision_matched"]["acc"]
            o = summary["decision"]["oracle"]["decision_matched"]["acc"]
            summary["deltas_matched"] = {
                "bind_minus_unguided": (b - u) if (b is not None and u is not None) else None,
                "oracle_minus_unguided": (o - u) if (o is not None and u is not None) else None,
                "oracle_minus_bind": (o - b) if (o is not None and b is not None) else None,
            }

        sum_p = args.dest / f"{slug}.summary.json"
        sum_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        def pct(d):
            a = (d or {}).get("acc")
            return "—" if a is None else f"{100 * a:.1f}"

        lines = [
            f"# ADS Method 1 bind n={len(rows)} — {slug}",
            "",
            "Primary = **Decision among matched gold spans**. Extra OCR ignored. Missing gold reported.",
            "",
            "| Cond | Decision@matched | missing% | coverage | extras/item |",
            "|---|---:|---:|---:|---:|",
        ]
        for cond in ("unguided", "bind_act", "oracle"):
            b = summary["decision"].get(cond)
            if not b:
                continue
            miss = b["missing"]["rate"]
            cov = b["coverage"]["rate"]
            ex = b["n_extra_ignored"]["mean"]
            lines.append(
                f"| {cond} | {pct(b['decision_matched'])} | "
                f"{'—' if miss is None else f'{100*miss:.0f}'} | "
                f"{'—' if cov is None else f'{100*cov:.0f}'} | "
                f"{'—' if ex is None else f'{ex:.1f}'} |"
            )
        if summary.get("deltas_matched"):
            d = summary["deltas_matched"]
            lines += [
                "",
                f"- Δ Decision@matched (bind−unguided) = {d['bind_minus_unguided']}",
                f"- Δ (oracle−unguided) = {d['oracle_minus_unguided']}",
                f"- Δ (oracle−bind) = {d['oracle_minus_bind']}",
                "",
                "Missing gold by taxonomy (bind): "
                + str(summary["decision"].get("bind_act", {}).get("missing", {}).get("by_taxonomy")),
            ]
        md = args.dest / f"{slug}.summary.md"
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        print("wrote", jsonl, sum_p, md)


if __name__ == "__main__":
    main()
