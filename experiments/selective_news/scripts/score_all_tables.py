#!/usr/bin/env python3
"""Aggregate paper tables from selective_news results/ into markdown + latex."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{100 * x:.1f}"
    return str(x)


def score_e1(results: Path, out: Path) -> str:
    lines = [
        "# Table E1 — KNOW-rules (verbal YES/NO)",
        "",
        "| Model | n_ok | n | acc% |",
        "|---|---:|---:|---:|",
    ]
    tex = [
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Verbal rule knowledge (YES/NO on constructional rules, blank image).}",
        "\\label{tab:know-rules}",
        "\\begin{tabular}{l r r r}\\toprule Model & $n_{ok}$ & $n$ & Acc. \\\\ \\midrule",
    ]
    for p in sorted((results / "rule_yesno").glob("*.json")):
        obj = load_json(p)
        if not obj:
            continue
        models = obj.get("models") or [obj]
        for m in models:
            mid = (m.get("model") or p.stem).split("/")[-1]
            n_ok, n = m.get("n_ok", 0), m.get("n", 0)
            acc = (n_ok / n) if n else None
            lines.append(f"| {mid} | {n_ok} | {n} | {fmt(acc)} |")
            tex.append(f"{mid.replace('_', r'_')} & {n_ok} & {n} & {fmt(acc)} \\\\")
    tex += ["\\bottomrule\\end{tabular}\\end{table}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_e1_know_rules.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "table_e1_know_rules.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return str(out / "table_e1_know_rules.md")


def score_tables(results: Path, out: Path) -> str:
    src = results / "news_tables_act"
    if not (src / "summary.json").exists():
        src = results / "news_tables_strict"
    if not (src / "summary.json").exists():
        src = results / "news_tables"
    summary = load_json(src / "summary.json")
    if not summary:
        # try run scorer
        return "missing news_tables/summary.json — run score_news_tables.py first"
    # copy existing paper tables
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tables.md", "table1.tex", "table3.tex", "summary.json"):
        p = src / name
        if p.exists():
            (out / f"paper_{name}").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    # also emit explicit E2–E8 column guide
    guide = [
        "# Experiment → column map (Table 1 / Table 3)",
        "",
        "| # | Experiment | Column |",
        "|---|---|---|",
        "| 1 | Rule KNOW | KNOW-rules |",
        "| 2 | KNOW spans (no image) | KNOW |",
        "| 3 | Decision (with image) | Decision |",
        "| 4 | Preserve | Preserve |",
        "| 5 | KNOW→ACT gap | KNOW→ACT |",
        "| 6 | Translation | Translation |",
        "| 7 | Generation (unguided) | Generation / Unguided |",
        "| 8 | Guided vs unguided | Table 3 Δ |",
        "",
    ]
    (out / "experiment_column_map.md").write_text("\n".join(guide), encoding="utf-8")
    return str(out / "paper_tables.md")


def score_method(results: Path, out: Path) -> str:
    dest = results / "bind_method"
    lines = [
        "# Table E10 — Method1 Bind-then-Act vs unguided / oracle",
        "",
        "| Model | n | unguided | bind | oracle | Δ(bind−unguided) | Δ(oracle−unguided) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(dest.glob("*.summary.json")) if dest.exists() else []:
        s = load_json(p)
        if not s:
            continue
        # smoke_bind_method1 writes per-model summary inside jsonl sidecar; also check *.json
        pass
    # parse from jsonl summaries written by smoke_bind_method1
    for p in sorted(dest.glob("*.json")) if dest.exists() else []:
        if p.name.endswith(".jsonl") or "know_rules" in p.name:
            continue
        obj = load_json(p)
        if not obj:
            continue
        if "unguided" in obj or "bind" in obj or "summary" in obj:
            s = obj.get("summary") or obj
            mid = p.stem
            lines.append(
                f"| {mid} | {s.get('n','—')} | {fmt(s.get('unguided'))} | {fmt(s.get('bind'))} | "
                f"{fmt(s.get('oracle'))} | {fmt(s.get('bind_minus_unguided'))} | "
                f"{fmt(s.get('oracle_minus_unguided'))} |"
            )
    # also look for metrics json produced at end of smoke_bind
    for p in sorted(dest.glob("*metrics*.json")) if dest.exists() else []:
        s = load_json(p) or {}
        mid = p.stem
        lines.append(
            f"| {mid} | {s.get('n','—')} | {fmt(s.get('unguided'))} | {fmt(s.get('bind'))} | "
            f"{fmt(s.get('oracle'))} | {fmt(s.get('bind_minus_unguided'))} | "
            f"{fmt(s.get('oracle_minus_unguided'))} |"
        )
    # Fallback: scan jsonl and compute crude means if summary files missing
    if len(lines) <= 4 and dest.exists():
        for jp in sorted(dest.glob("*.jsonl")):
            rows = [json.loads(x) for x in jp.read_text().splitlines() if x.strip()]
            if not rows:
                continue
            def mean(key):
                xs = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
                # nested scores
                if not xs:
                    xs = []
                    for r in rows:
                        for path in (
                            ("scores", key),
                            ("unguided", "acc"),
                            ("bind", "acc"),
                            ("oracle", "acc"),
                        ):
                            pass
                        sc = r.get("scores") or {}
                        if key in sc and isinstance(sc[key], (int, float)):
                            xs.append(sc[key])
                        elif isinstance(sc.get(key), dict) and "acc" in sc[key]:
                            xs.append(sc[key]["acc"])
                return (sum(xs) / len(xs)) if xs else None
            # try common field names from smoke_bind_method1
            u = b = o = None
            us = bs = os_ = []
            for r in rows:
                for block, bucket in (
                    ("S4_unguided", us),
                    ("unguided", us),
                    ("bind", bs),
                    ("S4_bind", bs),
                    ("oracle", os_),
                    ("S4_oracle", os_),
                ):
                    blk = r.get(block)
                    if isinstance(blk, dict) and "acc" in blk:
                        bucket.append(blk["acc"])
                    elif isinstance(blk, list) and blk:
                        ok = sum(1 for x in blk if x.get("ok"))
                        bucket.append(ok / len(blk))
            def avg(xs):
                return sum(xs) / len(xs) if xs else None
            u, b, o = avg(us), avg(bs), avg(os_)
            mid = jp.stem
            lines.append(
                f"| {mid} | {len(rows)} | {fmt(u)} | {fmt(b)} | {fmt(o)} | "
                f"{fmt((b-u) if u is not None and b is not None else None)} | "
                f"{fmt((o-u) if u is not None and o is not None else None)} |"
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_e10_method.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out / "table_e10_method.md")


def score_strict(results: Path, out: Path) -> str:
    dest = results / "strict_small"
    lines = ["# Strict-small (n≈25, miss=fail)", ""]
    if dest.exists():
        for p in sorted(dest.glob("*")):
            if p.suffix in (".md", ".tex", ".json"):
                lines.append(f"- `{p.name}`")
        # copy any summary
        for p in dest.glob("*summary*"):
            lines.append("")
            lines.append(p.read_text(encoding="utf-8")[:4000])
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_strict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out / "table_strict.md")


def score_lang(results: Path, out: Path) -> str:
    dest = results / "lang"
    lines = [
        "# Table E9 — Language pairs",
        "",
        "| Pair | Model | n | KNOW | Decision | Preserve | Gen | Guided | Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if dest.exists():
        for sub in sorted(dest.iterdir()):
            if not sub.is_dir():
                continue
            summary = load_json(sub / "summary.json")
            if not summary:
                continue
            for slug, s in summary.items():
                lines.append(
                    f"| {sub.name} | {slug} | {s.get('n_items')} | {fmt(s['know']['acc'])} | "
                    f"{fmt(s['decision']['acc'])} | {fmt(s['preserve']['acc'])} | "
                    f"{fmt(s['unguided']['acc'])} | {fmt(s['guided']['acc'])} | {fmt(s['delta'])} |"
                )
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_e9_lang.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out / "table_e9_lang.md")


def write_index(out: Path, written: list[str]) -> None:
    md = ["# Paper tables index", ""]
    for w in written:
        md.append(f"- `{w}`")
    md += [
        "",
        "## How to paste into the paper",
        "1. Table 1 / Table 3 → `paper_table1.tex`, `paper_table3.tex` (from news_tables).",
        "2. KNOW-rules → `table_e1_know_rules.tex`.",
        "3. Method intervention → `table_e10_method.md` (convert to tex as needed).",
        "4. Language appendix → `table_e9_lang.md`.",
    ]
    (out / "INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--which", default="all", choices=["e1", "tables", "method", "strict", "lang", "all"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.results_dir / "paper_tables")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    which = args.which
    if which in ("e1", "all"):
        written.append(score_e1(args.results_dir, out))
    if which in ("tables", "all"):
        written.append(score_tables(args.results_dir, out))
    if which in ("method", "all"):
        written.append(score_method(args.results_dir, out))
    if which in ("strict", "all"):
        written.append(score_strict(args.results_dir, out))
    if which in ("lang", "all"):
        written.append(score_lang(args.results_dir, out))
    write_index(out, [w for w in written if w])
    print("paper tables →", out)


if __name__ == "__main__":
    main()
