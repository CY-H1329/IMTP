#!/usr/bin/env python3
"""Strict scoring: missing gold span = fail; extras ignored."""
from __future__ import annotations

import json
import re
from typing import Callable

from oracle_guide import similar


def parse_loose(raw: str, parse_json_items: Callable) -> list[dict]:
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
            if out is None or out == "":
                ok = False
                reason = "preserve_no_output"
            elif similar(out, r["text"]) or similar(r["text"], str(out)):
                ok = True
                reason = "ok"
            else:
                ok = False
                reason = "preserve_rewritten"
        elif check_output and r["decision"] == "translate":
            # decision must be translate; if gt_tgt present, prefer matching it
            gt = r.get("gt_tgt") or ""
            matched_tgt = bool(gt) and out is not None and (
                similar(str(out), gt) or similar(gt, str(out))
            )
            kept_src = out is not None and (
                similar(str(out), r["text"]) or similar(r["text"], str(out))
            )
            if pred != "translate":
                ok = False
                reason = "wrong_decision"
            elif kept_src and not matched_tgt:
                ok = False
                reason = "translate_kept_source"
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
                    "matched_gt_tgt": matched_tgt,
                    "kept_source": kept_src,
                }
            )
            continue
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


def score_block_strict(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(1 for x in rows if x.get("ok"))
    miss = sum(1 for x in rows if x.get("missing"))
    return {
        "n": n,
        "ok": ok,
        "acc": (ok / n) if n else None,
        "missing": miss,
        "missing_rate": (miss / n) if n else None,
    }


def item_perfect(rows: list[dict]) -> bool:
    return bool(rows) and all(r.get("ok") for r in rows)


def fmt_progress(done: int, total: int, prefix: str = "") -> str:
    pct = (100.0 * done / total) if total else 100.0
    bar_n = 20
    filled = int(bar_n * done / total) if total else bar_n
    bar = "#" * filled + "-" * (bar_n - filled)
    head = f"{prefix} " if prefix else ""
    return f"{head}[{bar}] {done}/{total} ({pct:.1f}%)"
