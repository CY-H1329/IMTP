#!/usr/bin/env python3
"""Score news tables with Decision probe + ACT (preserve / translate@gt).

Paper-facing metrics:
  - Decision: alone, with image, gold spans listed (finding)
  - Preserve@Decision
  - ACT-Preserve unguided vs guided (did they keep PHOTO_SIGN etc.)
  - ACT-Translate unguided vs guided (output ≈ official gt_tgt)
  - Δ guided − unguided on ACT-P and ACT-T
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _sb() -> Path:
    e = os.environ.get("SELECTIVE_BENCH_ROOT")
    if e:
        return Path(e)
    here = Path(__file__).resolve().parent.parent
    if (here / "results").exists() or (here / "gold").exists():
        return here
    return Path("/workspace/chanyeong/ICLR/selective_bench")


ROOT = _sb()
DEST = Path(os.environ.get("NEWS_TABLES_DIR", ROOT / "results" / "news_tables_act"))


def acc(rows: list[dict], key: str = "ok") -> dict:
    n = len(rows)
    ok = sum(1 for x in rows if x.get(key))
    miss = sum(1 for x in rows if x.get("missing"))
    return {
        "n": n,
        "ok": ok,
        "acc": (ok / n) if n else None,
        "missing": miss,
        "missing_rate": (miss / n) if n else None,
    }


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
    pu = [x for x in s4u if x.get("expect") == "preserve"]
    pg = [x for x in s4g if x.get("expect") == "preserve"]
    tu = [x for x in s4u if x.get("expect") == "translate"]
    tg = [x for x in s4g if x.get("expect") == "translate"]

    def tgt_acc(xs):
        n = len(xs)
        ok = sum(1 for x in xs if x.get("matched_gt_tgt"))
        return {"n": n, "ok": ok, "acc": (ok / n) if n else None}

    au = acc(s4u)
    ag = acc(s4g)
    ap_u, ap_g = acc(pu), acc(pg)
    at_u, at_g = tgt_acc(tu), tgt_acc(tg)

    def delta(a, b):
        if a is None or b is None:
            return None
        return a - b

    # KNOW→ACT gap on preserve (knew at KNOW but failed ACT unguided)
    know_p_ok = {
        (r.get("id"), x["span"])
        for r in rows
        for x in r.get("know") or []
        if x.get("expect") == "preserve" and x.get("ok")
    }
    gap = n_gap = 0
    for r in rows:
        for x in r.get("S4_unguided") or []:
            if x.get("expect") != "preserve":
                continue
            n_gap += 1
            if (r.get("id"), x.get("span")) in know_p_ok and not x.get("ok"):
                gap += 1

    return {
        "n_items": len(rows),
        "scoring": "decision_probe+act_gt",
        "know": acc(know),
        "decision": acc(dec),
        "preserve_decision": acc(preserve_d),
        "act_preserve_unguided": ap_u,
        "act_preserve_guided": ap_g,
        "act_translate_unguided": at_u,
        "act_translate_guided": at_g,
        "delta_preserve": delta(ap_g["acc"], ap_u["acc"]),
        "delta_translate": delta(at_g["acc"], at_u["acc"]),
        "act_unguided": au,
        "act_guided": ag,
        "delta_act": delta(ag["acc"], au["acc"]),
        "know_to_act": {"n": n_gap, "gap": gap, "rate": (gap / n_gap) if n_gap else None},
        # aliases for older paper_tables helpers
        "preserve": acc(preserve_d),
        "translation": at_u,
        "generation": au,
        "unguided": au,
        "guided": ag,
        "delta": delta(ag["acc"], au["acc"]),
        "act": au,
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
    slugs = sorted(
        {
            p.name.split(".shard")[0].replace(".jsonl", "")
            for p in dest.glob("*.jsonl")
            if not p.name.startswith("_")
        }
    )
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

    md = [
        "# News selective translation — Decision + ACT",
        "",
        "Decision = alone with image (finding). Guided does **not** use Decision for the claim;",
        "guided ACT = execution given TRANSLATE/PRESERVE lists. Translate OK = output≈gt_tgt.",
        "",
        "## Table A. Finding (Decision probe)",
        "",
        "| Model | n | KNOW | Decision | Preserve@Dec |",
        "|---|---:|---:|---:|---:|",
    ]
    for slug, s in summaries.items():
        md.append(
            f"| {slug} | {s['n_items']} | {fmt(s['know']['acc'])} | "
            f"{fmt(s['decision']['acc'])} | {fmt(s['preserve_decision']['acc'])} |"
        )

    md += [
        "",
        "## Table B. ACT: unguided vs guided (what matters for Δ)",
        "",
        "| Model | ACT-P unguided | ACT-P guided | ΔP | ACT-T unguided | ACT-T guided | ΔT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for slug, s in summaries.items():
        md.append(
            f"| {slug} | {fmt(s['act_preserve_unguided']['acc'])} | "
            f"{fmt(s['act_preserve_guided']['acc'])} | {fmt(s['delta_preserve'])} | "
            f"{fmt(s['act_translate_unguided']['acc'])} | "
            f"{fmt(s['act_translate_guided']['acc'])} | {fmt(s['delta_translate'])} |"
        )

    tex_a = [
        "% Table Decision probe",
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Finding: KNOW (no image) and Decision (with image) on gold spans. "
        "Guided inventory is not used in this table.}",
        "\\label{tab:news-decision}",
        "\\begin{tabular}{l r r r r}\\toprule",
        "Model & n & KNOW & Decision & Pres.@Dec \\\\",
        "\\midrule",
    ]
    for slug, s in summaries.items():
        tex_a.append(
            f"{slug.replace('_', r'_')} & {s['n_items']} & {fmt(s['know']['acc'])} & "
            f"{fmt(s['decision']['acc'])} & {fmt(s['preserve_decision']['acc'])} \\\\"
        )
    tex_a += ["\\bottomrule\\end{tabular}\\end{table}", ""]

    tex_b = [
        "% Table ACT unguided vs guided",
        "\\begin{table}[t]\\centering\\small",
        "\\caption{ACT execution. Preserve: output$\\approx$source. Translate: output$\\approx$official target. "
        "Guided receives the TRANSLATE/PRESERVE lists (finding given); unguided must discover and act.}",
        "\\label{tab:news-act-guided}",
        "\\begin{tabular}{l r r r r r r}\\toprule",
        "Model & P-ung & P-g & $\\Delta$P & T-ung & T-g & $\\Delta$T \\\\",
        "\\midrule",
    ]
    for slug, s in summaries.items():
        tex_b.append(
            f"{slug.replace('_', r'_')} & {fmt(s['act_preserve_unguided']['acc'])} & "
            f"{fmt(s['act_preserve_guided']['acc'])} & {fmt(s['delta_preserve'])} & "
            f"{fmt(s['act_translate_unguided']['acc'])} & "
            f"{fmt(s['act_translate_guided']['acc'])} & {fmt(s['delta_translate'])} \\\\"
        )
    tex_b += ["\\bottomrule\\end{tabular}\\end{table}", ""]

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (dest / "table_decision.tex").write_text("\n".join(tex_a), encoding="utf-8")
    (dest / "table_act.tex").write_text("\n".join(tex_b), encoding="utf-8")
    # keep old names as copies for run_4gpu score helpers
    (dest / "table1.tex").write_text("\n".join(tex_a), encoding="utf-8")
    (dest / "table3.tex").write_text("\n".join(tex_b), encoding="utf-8")
    (dest / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest / "tables.md")
    print(
        f"{'model':28s} {'n':>5} {'Dec':>6} "
        f"{'P_ung':>6} {'P_g':>6} {'ΔP':>6} "
        f"{'T_ung':>6} {'T_g':>6} {'ΔT':>6}"
    )
    for slug, s in summaries.items():
        print(
            f"{slug[:28]:28s} {s['n_items']:5d} {fmt(s['decision']['acc']):>6} "
            f"{fmt(s['act_preserve_unguided']['acc']):>6} {fmt(s['act_preserve_guided']['acc']):>6} "
            f"{fmt(s['delta_preserve']):>6} "
            f"{fmt(s['act_translate_unguided']['acc']):>6} {fmt(s['act_translate_guided']['acc']):>6} "
            f"{fmt(s['delta_translate']):>6}"
        )
        # raw counts too
        print(
            f"{'':28s}       "
            f"P {s['act_preserve_unguided']['ok']}/{s['act_preserve_unguided']['n']} → "
            f"{s['act_preserve_guided']['ok']}/{s['act_preserve_guided']['n']}  |  "
            f"T(gt) {s['act_translate_unguided']['ok']}/{s['act_translate_unguided']['n']} → "
            f"{s['act_translate_guided']['ok']}/{s['act_translate_guided']['n']}"
        )


if __name__ == "__main__":
    main()
