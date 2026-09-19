#!/usr/bin/env python3
"""Build ICLR-style LaTeX tables from probe_chain_n25 JSONL."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

DIR = Path("/workspace/chanyeong/ICLR/selective_bench/results/probe_chain_n25")
OUT = DIR / "tables.tex"

DISPLAY = {
    "Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "InternVL2_5-8B": "InternVL2.5-8B",
}
STAGES = [
    ("S0_present", "S0$_{\\mathrm{pres}}$"),
    ("S0_identify", "S0$_{\\mathrm{id}}$"),
    ("S1_classify", "S1"),
    ("S2_policy", "S2"),
    ("S3_decide_image", "S3"),
    ("S4_generate", "S4"),
]
CATS = ["ads", "video", "news"]
DROPS = [
    ("S0found_S1", "S0$\\to$S1"),
    ("S1ok_S2", "S1$\\to$S2"),
    ("S2ok_S3", "S2$\\to$S3"),
    ("S3ok_S4", "S3$\\to$S4"),
]


def acc(rows, key="ok"):
    scored = [r for r in rows if r.get(key) is not None]
    n = len(scored)
    ok = sum(1 for r in scored if r.get(key))
    return n, ok, (ok / n) if n else None


def pct(x):
    return f"{100.0 * x:.1f}" if x is not None else "--"


def cell(n, ok, a):
    if a is None:
        return "--"
    return f"{pct(a)}"


def summarize(path: Path) -> dict:
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_stage = defaultdict(list)
    by_cat_stage = defaultdict(list)
    drop = defaultdict(lambda: defaultdict(int))
    for it in items:
        cat = it["category"]
        s0_by = {}
        for r in it.get("S0") or []:
            s0_by[r.get("rule_id") or r.get("taxonomy")] = r
            by_stage["S0_present"].append(r)
            by_cat_stage[(cat, "S0_present")].append(r)
            if r.get("identify_ok") is not None:
                by_stage["S0_identify"].append(r)
                by_cat_stage[(cat, "S0_identify")].append(r)
        grouped = defaultdict(dict)
        for r in it.get("span_probes") or []:
            key = r.get("rule_id") or r.get("taxonomy")
            grouped[key][r["stage"]] = r
            by_stage[r["stage"]].append(r)
            by_cat_stage[(cat, r["stage"])].append(r)
        s4 = {(r.get("rule_id") or r.get("taxonomy")): r for r in it.get("S4") or []}
        for key, st in grouped.items():
            s0 = s0_by.get(key)
            s1, s2, s3 = st.get("S1_classify"), st.get("S2_policy"), st.get("S3_decide_image")
            b = s4.get(key)
            if s0 and s1:
                s0_yes = s0.get("pred_present") == "YES" and bool(s0.get("span"))
                drop["S0found_S1"]["n"] += int(s0_yes)
                drop["S0found_S1"]["ok"] += int(s0_yes and bool(s1.get("ok")))
            if s1 and s2:
                drop["S1ok_S2"]["n"] += int(bool(s1.get("ok")))
                drop["S1ok_S2"]["ok"] += int(bool(s1.get("ok") and s2.get("ok")))
            if s2 and s3:
                drop["S2ok_S3"]["n"] += int(bool(s2.get("ok")))
                drop["S2ok_S3"]["ok"] += int(bool(s2.get("ok") and s3.get("ok")))
            if s3 and b:
                drop["S3ok_S4"]["n"] += int(bool(s3.get("ok")))
                drop["S3ok_S4"]["ok"] += int(bool(s3.get("ok") and b.get("ok")))
                by_stage["S4_generate"].append(b)
                by_cat_stage[(cat, "S4_generate")].append(b)
    return {
        "n_items": len(items),
        "by_stage": {k: acc(v, "identify_ok" if k == "S0_identify" else "present_ok" if k == "S0_present" else "ok") for k, v in by_stage.items()},
        "by_cat_stage": {
            (c, st): acc(rows, "identify_ok" if st == "S0_identify" else "present_ok" if st == "S0_present" else "ok")
            for (c, st), rows in by_cat_stage.items()
        },
        "drop": {
            k: (v["n"], v["ok"], (v["ok"] / v["n"]) if v["n"] else None) for k, v in drop.items()
        },
    }


def main() -> None:
    models = []
    data = {}
    for p in sorted(DIR.glob("*.jsonl")):
        name = DISPLAY.get(p.stem, p.stem)
        data[name] = summarize(p)
        models.append(name)

    lines = []
    lines.append("% Auto-generated from probe_chain_n25 (25 items × 3 categories). Isolated sessions.")
    lines.append("% S0_pres: presence YES/NO. S0_id: quoted span matches gold. S1 classify. S2 text policy. S3 image policy. S4 generation.")
    lines.append("")

    # Table 1: overall stages
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Probe-chain accuracy on the $n{=}25$ per-category pilot (75 images/model). Each stage is a fresh session with no chat history.}")
    lines.append("\\label{tab:probe-n25-stage}")
    lines.append("\\begin{tabular}{l" + "c" * len(STAGES) + "}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(lab for _, lab in STAGES) + " \\\\")
    lines.append("\\midrule")
    for m in models:
        cells = []
        for sid, _ in STAGES:
            n, ok, a = data[m]["by_stage"].get(sid, (0, 0, None))
            cells.append(cell(n, ok, a))
        lines.append(m + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2: by category, S0 and S4
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Accuracy by category on the $n{=}25$ pilot. S0$_{\\mathrm{pres}}$ is presence detection; S4 is whether the generation actually translated or preserved the S0 span.}")
    lines.append("\\label{tab:probe-n25-cat}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Model & Split & S0$_{\\mathrm{pres}}$ & S2 & S3 & S4 \\\\")
    lines.append("\\midrule")
    for m in models:
        for i, cat in enumerate(CATS):
            row = [m if i == 0 else "", cat]
            for st in ("S0_present", "S2_policy", "S3_decide_image", "S4_generate"):
                n, ok, a = data[m]["by_cat_stage"].get((cat, st), (0, 0, None))
                row.append(cell(n, ok, a))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 3: dropoff
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Conditional accuracy along the chain: given the previous stage was correct, does the next isolated session still get it right?}")
    lines.append("\\label{tab:probe-n25-drop}")
    lines.append("\\begin{tabular}{l" + "c" * len(DROPS) + "}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(lab for _, lab in DROPS) + " \\\\")
    lines.append("\\midrule")
    for m in models:
        cells = []
        for did, _ in DROPS:
            n, ok, a = data[m]["drop"].get(did, (0, 0, None))
            cells.append(cell(n, ok, a) if n else "--")
        lines.append(m + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # counts footnote table
    lines.append("% n/ok counts")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Raw $ok/n$ for each stage on the $n{=}25$ pilot.}")
    lines.append("\\label{tab:probe-n25-counts}")
    lines.append("\\begin{tabular}{l" + "c" * len(STAGES) + "}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(lab for _, lab in STAGES) + " \\\\")
    lines.append("\\midrule")
    for m in models:
        cells = []
        for sid, _ in STAGES:
            n, ok, a = data[m]["by_stage"].get(sid, (0, 0, None))
            cells.append(f"{ok}/{n}" if n else "--")
        lines.append(m + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)
    for m in models:
        print(m, {k: data[m]["by_stage"].get(k) for k, _ in STAGES})


if __name__ == "__main__":
    main()
