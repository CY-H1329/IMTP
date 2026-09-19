#!/usr/bin/env python3
"""Score news Table 1 + Table 3 JSONL into markdown/latex."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import os

def _sb() -> Path:
    e = os.environ.get("SELECTIVE_BENCH_ROOT")
    if e:
        return Path(e)
    here = Path(__file__).resolve().parent.parent
    if (here / "results").exists() or (here / "gold").exists():
        return here
    return Path("/workspace/chanyeong/ICLR/selective_bench")

ROOT = _sb()
DEST = Path(os.environ.get("NEWS_TABLES_DIR", ROOT / "results" / "news_tables"))


def acc(rows: list[dict], key: str = "ok") -> dict:
    n = len(rows)
    ok = sum(1 for x in rows if x.get(key))
    return {"n": n, "ok": ok, "acc": (ok / n) if n else None}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        i = rec.get("id")
        if i in seen:
            continue
        seen.add(i)
        rows.append(rec)
    return rows


def merge_shards(dest: Path, slug: str) -> list[dict]:
    files = sorted(dest.glob(f"{slug}.shard*.jsonl")) + sorted(dest.glob(f"{slug}.jsonl"))
    rows, seen = [], set()
    for p in files:
        for rec in load_jsonl(p):
            i = rec["id"]
            if i in seen:
                continue
            seen.add(i)
            rows.append(rec)
    return rows


def summarize(rows: list[dict]) -> dict:
    know = [x for r in rows for x in r.get("know") or []]
    dec = [x for r in rows for x in r.get("decision") or []]
    s4u = [x for r in rows for x in r.get("S4_unguided") or []]
    s4g = [x for r in rows for x in r.get("S4_guided") or []]
    preserve_d = [x for x in dec if x.get("expect") == "preserve"]
    preserve_a = [x for x in s4u if x.get("expect") == "preserve"]
    trans_a = [x for x in s4u if x.get("expect") == "translate"]
    trans_ok_tgt = [x for x in trans_a if x.get("matched_gt_tgt")]
    photo = [x for x in s4u if x.get("taxonomy") == "PHOTO_SIGN"]
    # KNOW→ACT: gold-P known as preserve but rewritten in generation
    know_p_ok = {(r.get("id"), x["span"]) for r in rows for x in r.get("know") or [] if x.get("expect") == "preserve" and x.get("ok")}
    act_p_bad = 0
    act_p_n = 0
    for r in rows:
        for x in r.get("S4_unguided") or []:
            if x.get("expect") != "preserve":
                continue
            act_p_n += 1
            if (r.get("id"), x.get("span")) in know_p_ok and not x.get("ok"):
                act_p_bad += 1
    u = acc(s4u)
    g = acc(s4g)
    return {
        "n_items": len(rows),
        "know": acc(know),
        "decision": acc(dec),
        "preserve": acc(preserve_d),
        "act": acc(s4u),
        "preserve_act": acc(preserve_a),
        "translation": {"n": len(trans_a), "ok": len(trans_ok_tgt), "acc": (len(trans_ok_tgt) / len(trans_a)) if trans_a else None},
        "generation": u,
        "photo_sign": acc(photo),
        "know_to_act": {"n": act_p_n, "gap": act_p_bad, "rate": (act_p_bad / act_p_n) if act_p_n else None},
        "unguided": u,
        "guided": g,
        "delta": (g["acc"] - u["acc"]) if (g["acc"] is not None and u["acc"] is not None) else None,
    }


def fmt(x) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dest-dir", type=Path, default=DEST)
    dest = ap.parse_args().dest_dir
    slugs = sorted({p.name.split(".shard")[0].replace(".jsonl", "") for p in dest.glob("*.jsonl") if not p.name.startswith("_")})
    slugs = [s for s in slugs if s]
    summaries = {}
    for slug in slugs:
        rows = merge_shards(dest, slug)
        if not rows:
            continue
        summaries[slug] = summarize(rows)
        kr = dest / f"{slug}.know_rules.json"
        if kr.exists():
            summaries[slug]["know_rules"] = json.loads(kr.read_text())
    md = ["# News 3000 — Table 1 and Table 3", ""]
    md += [
        "## Table 1. KNOW / Decision / Preserve / ACT / Translation / Generation",
        "",
        "| Model | n | KNOW-rules | KNOW | Decision | Preserve | ACT | Translation | Generation | KNOW→ACT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex1 = [
        "% Table 1 news_eval",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Selective translation on the news gold. KNOW-rules = verbal yes/no on N1/N2. KNOW = T/P on gold spans without the image. Decision = T/P with the image. Preserve = gold-P spans. ACT = unguided generation follows T/P. Translation = gold-T span matches the official target. Generation = overall unguided S4. KNOW$\\rightarrow$ACT = gold-P correctly listed at KNOW but rewritten in generation.}",
        "\\label{tab:news-know-act}",
        "\\begin{tabular}{l r r r r r r r r}",
        "\\toprule",
        "Model & n & KNOW-R & KNOW & Dec. & Pres. & ACT & Trans. & Gen. \\\\",
        "\\midrule",
    ]
    for slug, s in summaries.items():
        kr = s.get("know_rules") or {}
        kr_acc = (kr["n_ok"] / kr["n"]) if kr.get("n") else None
        md.append(
            f"| {slug} | {s['n_items']} | {fmt(kr_acc)} | {fmt(s['know']['acc'])} | "
            f"{fmt(s['decision']['acc'])} | {fmt(s['preserve']['acc'])} | "
            f"{fmt(s['act']['acc'])} | {fmt(s['translation']['acc'])} | "
            f"{fmt(s['generation']['acc'])} | {fmt(s['know_to_act']['rate'])} |"
        )
        slug_tex = slug.replace("_", r"\_")
        tex1.append(
            f"{slug_tex} & {s['n_items']} & {fmt(kr_acc)} & {fmt(s['know']['acc'])} & "
            f"{fmt(s['decision']['acc'])} & {fmt(s['preserve']['acc'])} & {fmt(s['act']['acc'])} & "
            f"{fmt(s['translation']['acc'])} & {fmt(s['generation']['acc'])} \\\\"
        )
    tex1 += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    md += [
        "",
        "## Table 3. Guided vs unguided S4",
        "",
        "| Model | n | Unguided | Guided | $\\Delta$ |",
        "|---|---:|---:|---:|---:|",
    ]
    tex3 = [
        "% Table 3 guided vs unguided",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Generation accuracy (S4) on the news gold inventory. Unguided: localize with no span list. Guided: rules + proper-noun identity + exact TRANSLATE/PRESERVE list (no gold target strings). Same images.}",
        "\\label{tab:news-guided-vs-unguided}",
        "\\begin{tabular}{l r r r r}",
        "\\toprule",
        "Model & n & Unguided & Guided & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for slug, s in summaries.items():
        d = s["delta"]
        md.append(f"| {slug} | {s['n_items']} | {fmt(s['unguided']['acc'])} | {fmt(s['guided']['acc'])} | {fmt(d)} |")
        slug_tex = slug.replace("_", r"\_")
        tex3.append(
            f"{slug_tex} & {s['n_items']} & {fmt(s['unguided']['acc'])} & "
            f"{fmt(s['guided']['acc'])} & {fmt(d)} \\\\"
        )
    tex3 += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (dest / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (dest / "table1.tex").write_text("\n".join(tex1), encoding="utf-8")
    (dest / "table3.tex").write_text("\n".join(tex3), encoding="utf-8")
    (dest / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest / "tables.md")
    for slug, s in summaries.items():
        print(slug, "n", s["n_items"], "unguided", s["unguided"]["acc"], "guided", s["guided"]["acc"], "delta", s["delta"])


if __name__ == "__main__":
    main()
