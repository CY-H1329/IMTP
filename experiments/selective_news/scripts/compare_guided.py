#!/usr/bin/env python3
"""Compare unguided vs guided-oracle S4 on the same gold spans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/workspace/chanyeong/ICLR/selective_bench")
DEFAULT_UNG = ROOT / "results" / "probe_chain_n25_recent"
DEFAULT_GUD = ROOT / "results" / "probe_chain_n25_guided_recent"

DISPLAY = {
    "Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "InternVL3_5-8B-HF": "InternVL3.5-8B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "InternVL2_5-8B": "InternVL2.5-8B",
}


def load(p: Path) -> dict:
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["id"]] = rec
    return out


def s4_rows(rec: dict, guided: bool) -> list[dict]:
    if guided:
        return rec.get("S4") or []
    return rec.get("S4_gold") or rec.get("S4") or []


def acc(rows: list[dict], guided: bool) -> dict:
    items = [x for r in rows for x in s4_rows(r, guided)]
    n = len(items)
    ok = sum(1 for x in items if x.get("ok"))
    return {"n": n, "ok": ok, "acc": (ok / n) if n else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ung", type=Path, default=DEFAULT_UNG)
    ap.add_argument("--gud", type=Path, default=DEFAULT_GUD)
    args = ap.parse_args()
    ung, gud = args.ung, args.gud
    outp = gud / "compare.json"
    tex = gud / "compare.tex"
    summary = {}
    lines = [
        "% Guided vs unguided S4, same gold spans, n=25 per category.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Generation accuracy (S4) on the gold inventory. Unguided: localize with no hint. Guided: rules + proper-noun identity + exact TRANSLATE/PRESERVE list (no gold targets). Same 75 images.}",
        "\\label{tab:guided-vs-unguided}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & Unguided & Guided & $\\Delta$ & ads $\\Delta$ & video $\\Delta$ & news $\\Delta$ \\\\",
        "\\midrule",
    ]
    for f in sorted(gud.glob("*.jsonl")):
        name = DISPLAY.get(f.stem, f.stem)
        g = load(f)
        u = load(ung / f.name)
        if not g:
            continue
        shared_g = [g[i] for i in g if i in u]
        shared_u = [u[i] for i in g if i in u]
        ua = acc(shared_u, guided=False)
        ga = acc(shared_g, guided=True)
        delta = (ga["acc"] - ua["acc"]) if ua["acc"] is not None and ga["acc"] is not None else None
        by = {}
        for cat in ("ads", "video", "news"):
            ids = [i for i, r in g.items() if r.get("category") == cat and i in u]
            by[cat] = {
                "unguided": acc([u[i] for i in ids], guided=False),
                "guided": acc([g[i] for i in ids], guided=True),
            }
            if by[cat]["unguided"]["acc"] is not None and by[cat]["guided"]["acc"] is not None:
                by[cat]["delta"] = by[cat]["guided"]["acc"] - by[cat]["unguided"]["acc"]
            else:
                by[cat]["delta"] = None
        summary[name] = {"n_items": len(shared_g), "unguided": ua, "guided": ga, "delta": delta, "by_category": by}

        def pct(x):
            return f"{100 * x:.1f}" if x is not None else "--"

        dlt = pct(delta) if delta is not None else "--"
        lines.append(
            f"{name} & {pct(ua['acc'])} & {pct(ga['acc'])} & {dlt} & "
            f"{pct(by['ads']['delta'])} & {pct(by['video']['delta'])} & {pct(by['news']['delta'])} \\\\"
        )
        print(name, "unguided", ua, "guided", ga, "delta", delta)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    gud.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", outp, tex)


if __name__ == "__main__":
    main()
