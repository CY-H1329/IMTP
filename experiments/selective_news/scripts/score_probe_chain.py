#!/usr/bin/env python3
"""Aggregate probe-chain JSONL: S0 present/identify, S1–S4, drop-off along the chain."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace/chanyeong/ICLR/selective_bench")
DEFAULT_DIR = ROOT / "results" / "probe_chain"


def acc(rows: list[dict], key: str = "ok") -> dict:
    scored = [r for r in rows if r.get(key) is not None]
    n = len(scored)
    ok = sum(1 for r in scored if r.get(key))
    return {"n": n, "ok": ok, "acc": (ok / n) if n else None}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()
    DIR = args.dir
    files = sorted(DIR.glob("*.jsonl"))
    if not files:
        print("no jsonl in", DIR)
        return
    out = {}
    for p in files:
        items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_stage = defaultdict(list)
        by_cat_stage = defaultdict(list)
        drop = defaultdict(lambda: defaultdict(int))
        for it in items:
            cat = it["category"]
            s0_by_tax = {}
            for r in it.get("S0") or []:
                s0_by_tax[r.get("rule_id") or r.get("taxonomy")] = r
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
            for r in it.get("S4_gold") or []:
                by_stage["S4_gold"].append(r)
                by_cat_stage[(cat, "S4_gold")].append(r)
            for key, st in grouped.items():
                s0 = s0_by_tax.get(key)
                s1, s2, s3 = st.get("S1_classify"), st.get("S2_policy"), st.get("S3_decide_image")
                b = s4.get(key)
                if s0 and s1:
                    s0_yes = s0.get("pred_present") == "YES" and bool(s0.get("span"))
                    drop["S0found_S1"]["n"] += int(s0_yes)
                    drop["S0found_S1"]["ok"] += int(s0_yes and bool(s1.get("ok")))
                    drop["S0ok_S1"]["n"] += int(bool(s0.get("present_ok")))
                    drop["S0ok_S1"]["ok"] += int(bool(s0.get("present_ok") and s1.get("ok")))
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
        summary = {
            "file": str(p),
            "n_items": len(items),
            "stage": {
                k: acc(v, "identify_ok" if k == "S0_identify" else "present_ok" if k == "S0_present" else "ok")
                for k, v in by_stage.items()
            },
            "by_category": {
                f"{c}/{st}": acc(
                    rows,
                    "identify_ok" if st == "S0_identify" else "present_ok" if st == "S0_present" else "ok",
                )
                for (c, st), rows in sorted(by_cat_stage.items())
            },
            "dropoff": {
                k: {
                    "n": v["n"],
                    "ok": v["ok"],
                    "acc": (v["ok"] / v["n"]) if v["n"] else None,
                }
                for k, v in drop.items()
            },
        }
        out[p.stem] = summary
        print(p.name, "items", len(items))
        for k, v in summary["stage"].items():
            print(f"  {k:20s} {v['ok']}/{v['n']} acc={v['acc']}")
        print("  dropoff", summary["dropoff"])
    (DIR / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", DIR / "summary.json")


if __name__ == "__main__":
    main()
