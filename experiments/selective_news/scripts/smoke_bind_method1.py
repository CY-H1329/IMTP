#!/usr/bin/env python3
"""Method 1 Bind-then-Act vs unguided vs oracle (n≈50).

Same image input for all conditions. Scoring:
  - gen_blob : legacy score_gen on raw string (may be noisy for JSON)
  - decision : match gold spans to parsed JSON decisions (primary metric)
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
GOLD = ROOT / "gold" / "news_eval.json"
DEST = ROOT / "results" / "bind_n50"


def unguided_prompt(item: dict) -> str:
    tgt_n = TGT_NAME.get(item.get("tgt_lang") or "ko", "Korean")
    return PROTO["prompts"]["S4_generate"].format(tgt_n=tgt_n)


def bind_act_prompt(item: dict) -> str:
    tgt_n = TGT_NAME.get(item.get("tgt_lang") or "ko", "Korean")
    return (
        f"Localize this news page image into {tgt_n}.\n\n"
        "You MUST follow Bind-then-Act. Do not jump to the final string.\n\n"
        f"RULES (constructional):\n{rules_block('news')}\n\n"
        "TYPE vocabulary (pick one per span):\n"
        "- headline / body / standfirst  → editorial CMS text\n"
        "- masthead                      → newspaper wordmark\n"
        "- photo_sign                    → diegetic lettering inside the photograph "
        "(building sign, plaque, screen, banner in the photo world)\n"
        "- person / org                  → proper names\n"
        "- other\n\n"
        "VISUAL_CONTEXT vocabulary (pick one):\n"
        "- cms_text_layer     → headline/body laid out as article chrome (not in the photo)\n"
        "- masthead_chrome    → site/newspaper wordmark area\n"
        "- on_photograph      → printed/painted text that lives inside the photo scene\n"
        "- caption_overlay    → burned-in caption over the image\n"
        "- uncertain\n\n"
        "For EVERY visible text span, output ONE JSON object with fields IN THIS ORDER:\n"
        "  text, type, visual_context, rule, decision, output\n"
        "decision MUST follow from type+visual_context+rule "
        "(photo_sign / on_photograph → preserve; "
        "headline/body/masthead / cms_text_layer → translate).\n"
        "Prioritize: main headline, masthead, body/standfirst, on-photo signs. "
        "Skip tiny UI chrome (font buttons, share icons) if needed for space.\n"
        "Return a JSON list only. No markdown fences."
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
    rows = []
    for r in inv:
        pred = None
        typ = vc = rule = out = None
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
                "ok": pred == r["decision"] if pred else False,
                "matched": pred is not None,
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


def subset(rows, tax=None, expect=None):
    out = rows
    if tax:
        out = [r for r in out if r.get("taxonomy") == tax]
    if expect:
        out = [r for r in out if r.get("expect") == expect]
    return out


def pick_items(gold: dict, n: int) -> list[dict]:
    """Half Dong-A, half Chosun when possible; require PHOTO_SIGN + translate."""
    donga, chosun = [], []
    for it in gold["items"]:
        inv = inventory(it)
        taxes = {r["taxonomy"] for r in inv}
        if "PHOTO_SIGN" not in taxes:
            continue
        if not any(r["decision"] == "translate" for r in inv):
            continue
        if not resolve_image(it["image"], it).exists():
            continue
        src = (it.get("source") or "").lower()
        if src == "chosun" or "chosun" in (it.get("article_id") or ""):
            chosun.append(it)
        else:
            donga.append(it)
    n_d = min(len(donga), (n + 1) // 2)
    n_c = min(len(chosun), n - n_d)
    # if chosun short, fill from donga
    out = donga[:n_d] + chosun[:n_c]
    if len(out) < n:
        rest = [x for x in donga[n_d:] + chosun[n_c:] if x not in out]
        out.extend(rest[: n - len(out)])
    return out[:n]


def load_done(path: Path) -> dict:
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        done[rec["id"]] = rec
    return done


def pack_rows(rows_list: list[list[dict]]) -> dict:
    all_r = [x for rows in rows_list for x in rows]
    matched = [x for x in all_r if x.get("matched")]
    return {
        "overall": acc(all_r),
        "overall_matched_only": acc(matched) if matched else {"n": 0, "ok": 0, "acc": None},
        "preserve": acc(subset(all_r, expect="preserve")),
        "photo_sign": acc(subset(all_r, tax="PHOTO_SIGN")),
        "translate": acc(subset(all_r, expect="translate")),
        "coverage": {
            "n": len(all_r),
            "matched": sum(1 for x in all_r if x.get("matched")),
            "rate": (sum(1 for x in all_r if x.get("matched")) / len(all_r)) if all_r else None,
        },
    }


def summarize(rows: list[dict], with_oracle: bool) -> dict:
    out = {
        "n_items": len(rows),
        "decision": {
            "unguided": pack_rows([r["dec_unguided"] for r in rows]),
            "bind_act": pack_rows([r["dec_bind"] for r in rows]),
        },
        "gen_blob": {
            "unguided": {
                "overall": acc([x for r in rows for x in r["unguided"]]),
                "photo_sign": acc(subset([x for r in rows for x in r["unguided"]], tax="PHOTO_SIGN")),
            },
            "bind_act": {
                "overall": acc([x for r in rows for x in r["bind_act"]]),
                "photo_sign": acc(subset([x for r in rows for x in r["bind_act"]], tax="PHOTO_SIGN")),
            },
        },
    }
    if with_oracle:
        out["decision"]["oracle"] = pack_rows([r["dec_oracle"] for r in rows])
        out["gen_blob"]["oracle"] = {
            "overall": acc([x for r in rows for x in r["oracle"]]),
            "photo_sign": acc(subset([x for r in rows for x in r["oracle"]], tax="PHOTO_SIGN")),
        }
        u = out["decision"]["unguided"]["overall"]["acc"]
        b = out["decision"]["bind_act"]["overall"]["acc"]
        o = out["decision"]["oracle"]["overall"]["acc"]
        out["deltas"] = {
            "bind_minus_unguided": (b - u) if (b is not None and u is not None) else None,
            "oracle_minus_unguided": (o - u) if (o is not None and u is not None) else None,
            "oracle_minus_bind": (o - b) if (o is not None and b is not None) else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3vl"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--max-side", type=int, default=512, help="image max side (512 safer on tight VRAM)")
    args = ap.parse_args()

    gold = json.loads(GOLD.read_text())
    items = pick_items(gold, args.n)
    args.dest.mkdir(parents=True, exist_ok=True)
    print(
        f"bind_n50 n={len(items)} oracle={not args.skip_oracle} dest={args.dest} "
        f"sources={[it.get('source') for it in items[:3]]}...",
        flush=True,
    )

    for spec in args.models:
        mid = ALIASES.get(spec, spec)
        slug = mid.split("/")[-1].replace(" ", "_")
        jsonl = args.dest / f"{slug}.bind_n50.jsonl"
        done = load_done(jsonl)
        print(f"resume done={len(done)}/{len(items)}", flush=True)

        vlm = LocalVLM(mid, max_side=args.max_side, max_new=900)
        # KNOW-rules once
        rules_p = args.dest / f"{slug}.know_rules.json"
        if not rules_p.exists():
            from run_rule_yesno import all_rules, fill_prompt, parse_yn  # noqa: E402
            from PIL import Image

            blank = ROOT / "results" / "_blank.png"
            blank.parent.mkdir(parents=True, exist_ok=True)
            if not blank.exists():
                Image.new("RGB", (448, 448), "white").save(blank)
            recs = []
            for r in all_rules():
                if r.get("split") != "news":
                    continue
                raw = vlm.generate(blank, fill_prompt(r), max_new=16)
                yn = parse_yn(raw)
                recs.append({"id": r["id"], "expect": r["expect"], "pred": yn, "ok": yn == r["expect"]})
            rules_p.write_text(json.dumps({"n_ok": sum(x["ok"] for x in recs), "n": len(recs), "items": recs}, indent=2))
            print(f"KNOW-rules {sum(x['ok'] for x in recs)}/{len(recs)}", flush=True)

        with jsonl.open("a", encoding="utf-8") as f:
            for i, it in enumerate(items):
                if it["id"] in done:
                    continue
                inv = inventory(it)
                img = resolve_image(it["image"], it)
                print(f"\n=== [{i+1}/{len(items)}] {slug} {it['id']} spans={len(inv)} ===", flush=True)

                raw_u = vlm.generate(img, unguided_prompt(it), max_new=450)
                raw_b = vlm.generate(img, bind_act_prompt(it), max_new=750)
                preds_u = parse_loose(raw_u or "")
                preds_b = parse_loose(raw_b or "")

                rec = {
                    "id": it["id"],
                    "article_id": it.get("article_id"),
                    "source": it.get("source"),
                    "image": str(img),
                    "src_lang": it.get("src_lang"),
                    "tgt_lang": it.get("tgt_lang"),
                    "unguided": score_gen(inv, raw_u or ""),
                    "bind_act": score_gen(inv, raw_b or ""),
                    "dec_unguided": match_decisions(inv, preds_u),
                    "dec_bind": match_decisions(inv, preds_b),
                    "bind_parsed": [
                        {
                            "text": (x.get("text") or "")[:120],
                            "type": x.get("type"),
                            "visual_context": x.get("visual_context"),
                            "decision": x.get("decision"),
                            "output": (x.get("output") or "")[:100],
                        }
                        for x in preds_b[:16]
                    ],
                    "raw": {
                        "unguided": (raw_u or "")[:3000],
                        "bind_act": (raw_b or "")[:4000],
                    },
                }
                if not args.skip_oracle:
                    raw_o = vlm.generate(img, guided_prompt(it, inv), max_new=450)
                    preds_o = parse_loose(raw_o or "")
                    rec["oracle"] = score_gen(inv, raw_o or "")
                    rec["dec_oracle"] = match_decisions(inv, preds_o)
                    rec["raw"]["oracle"] = (raw_o or "")[:3000]

                def fmt(name, dec):
                    a = acc(dec)
                    ph = acc(subset(dec, tax="PHOTO_SIGN"))
                    cov = sum(1 for x in dec if x.get("matched"))
                    print(
                        f"  {name:10s} dec={a['ok']}/{a['n']} "
                        f"PHOTO={ph['ok']}/{ph['n']} matched={cov}/{len(dec)}",
                        flush=True,
                    )

                fmt("unguided", rec["dec_unguided"])
                fmt("bind_act", rec["dec_bind"])
                if "dec_oracle" in rec:
                    fmt("oracle", rec["dec_oracle"])

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[it["id"]] = rec

        vlm.close()
        rows = [done[it["id"]] for it in items if it["id"] in done]
        summary = summarize(rows, with_oracle=not args.skip_oracle)
        sum_p = args.dest / f"{slug}.summary.json"
        sum_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        def pct(d):
            a = (d or {}).get("acc")
            return "—" if a is None else f"{100 * a:.1f}"

        lines = [
            f"# Method 1 bind n={len(rows)} — {slug}",
            "",
            "Primary metric = **decision** accuracy (parsed JSON vs gold spans).",
            "",
            "| Cond | overall | PHOTO_SIGN | preserve | translate | coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cond in ("unguided", "bind_act", "oracle"):
            b = summary["decision"].get(cond)
            if not b:
                continue
            cov = b["coverage"]
            cov_s = f"{100 * cov['rate']:.0f}%" if cov.get("rate") is not None else "—"
            lines.append(
                f"| {cond} | {pct(b['overall'])} | {pct(b['photo_sign'])} | "
                f"{pct(b['preserve'])} | {pct(b['translate'])} | {cov_s} |"
            )
        if summary.get("deltas"):
            d = summary["deltas"]
            lines += [
                "",
                f"- Δ(bind−unguided) = {d['bind_minus_unguided']}",
                f"- Δ(oracle−unguided) = {d['oracle_minus_unguided']}",
                f"- Δ(oracle−bind) = {d['oracle_minus_bind']}",
            ]
        md = args.dest / f"{slug}.summary.md"
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        print("wrote", jsonl, sum_p, md)


if __name__ == "__main__":
    main()
