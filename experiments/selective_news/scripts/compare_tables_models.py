#!/usr/bin/env python3
"""Compare models in news_tables_act/summary.json -> markdown."""
from __future__ import annotations
import json
import sys
from pathlib import Path

def pct(x):
    if x is None: return "—"
    if isinstance(x, dict):
        a = x.get("acc")
        return f"{100*a:.1f}" if a is not None else "—"
    return f"{100*x:.1f}" if isinstance(x, float) else str(x)

def main():
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "results/news_tables_act")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else dest / "compare_models.md")
    s = json.loads((dest / "summary.json").read_text())
    lines = ["# Model comparison (news_tables_act)", ""]
    lines.append("| Model | n | KNOW | Dec | DecI | Preserve@Dec | ACT_u | ACT_g | Δ | P-ung | P-g | T-ung | T-g |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m, v in sorted(s.items()):
        lines.append(
            f"| {m} | {v.get('n_items')} | {pct(v.get('know'))} | {pct(v.get('decision'))} | "
            f"{pct((v.get('decision_item') or {}).get('acc') if isinstance(v.get('decision_item'), dict) else v.get('decision_item'))} | "
            f"{pct(v.get('preserve'))} | {pct(v.get('unguided'))} | {pct(v.get('guided'))} | "
            f"{(v.get('delta') if v.get('delta') is not None else 0)*100:.1f} | "
            f"— | — | — | — |"
        )
    # Prefer tables.md metrics if present
    tm = dest / "tables.md"
    if tm.exists():
        lines += ["", "## Raw tables.md", "", tm.read_text()]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text())

if __name__ == "__main__":
    main()
