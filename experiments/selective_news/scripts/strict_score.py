#!/usr/bin/env python3
"""Evaluation for selective news.

Two different questions (do not mix scorers):

1) Decision probe (KNOW / Decision columns)
   - Model answers TRANSLATE|PRESERVE for listed gold spans.
   - Guided inventory is NOT used here — this measures finding alone.
   - Score: pred decision == gold decision. Missing span = fail.

2) ACT / generation (unguided vs guided)
   - Unguided: localize freely — did it *do* the right thing on each gold span?
   - Guided: we tell TRANSLATE/PRESERVE lists — Decision≈given; score *execution*.
   - Preserve OK ⇔ output ≈ source text
   - Translate OK ⇔ output ≈ official gt_tgt  (keeping source = FAIL; no free pass)
   - Missing gold span in JSON = FAIL
"""
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


def _find_pred(r: dict, preds: list[dict]) -> tuple[str | None, str | None]:
    pred = None
    out = None
    for p in preds:
        txt = p.get("text") or p.get("span") or ""
        if similar(txt, r["text"]) or similar(r["text"], txt):
            d = (p.get("decision") or "").lower()
            if d in ("translate", "preserve"):
                pred = d
            out = p.get("output")
            if out is None and p.get("text") and d == "preserve":
                out = p.get("text")
            break
    return pred, (None if out is None else str(out))


def score_decision_probe(inv: list[dict], preds: list[dict]) -> list[dict]:
    """Probe only: did the model label each gold span correctly?"""
    rows = []
    for r in inv:
        pred, _out = _find_pred(r, preds)
        if pred is None:
            ok, reason = False, "missing"
        elif pred != r["decision"]:
            ok, reason = False, "wrong_decision"
        else:
            ok, reason = True, "ok"
        rows.append(
            {
                "taxonomy": r["taxonomy"],
                "span": r["text"][:200],
                "expect": r["decision"],
                "pred": pred,
                "matched": pred is not None,
                "missing": pred is None,
                "ok": ok,
                "reason": reason,
                "metric": "decision_probe",
            }
        )
    return rows


def score_act(inv: list[dict], preds: list[dict], raw: str = "") -> list[dict]:
    """Generation ACT: preserve/translate quality vs gold expect + gt_tgt.

    Translate requires matching gt_tgt when available — absence of source in
    the blob is NOT enough for OK (fixes soft score_gen artefact).
    """
    rows = []
    raw = raw or ""
    for r in inv:
        expect = r["decision"]
        src = r["text"]
        gt = r.get("gt_tgt") or ""
        pred, out = _find_pred(r, preds)

        if pred is None and out is None:
            rows.append(
                {
                    "taxonomy": r["taxonomy"],
                    "span": src[:200],
                    "expect": expect,
                    "pred": None,
                    "output": None,
                    "matched": False,
                    "missing": True,
                    "ok": False,
                    "reason": "missing",
                    "matched_gt_tgt": False,
                    "kept_source": False,
                    "metric": "act",
                }
            )
            continue

        kept_source = bool(out) and (similar(out, src) or similar(src, out))
        matched_tgt = bool(gt) and bool(out) and (similar(out, gt) or similar(gt, out))

        if expect == "preserve":
            if not out:
                ok, reason = False, "preserve_no_output"
            elif kept_source:
                ok, reason = True, "ok"
            else:
                ok, reason = False, "preserve_rewritten"
        else:  # translate
            if not out:
                ok, reason = False, "translate_no_output"
            elif not gt:
                # no official target: must not keep source, must emit something else
                if kept_source:
                    ok, reason = False, "translate_kept_source"
                else:
                    ok, reason = True, "ok_no_gt"
            elif matched_tgt:
                ok, reason = True, "ok"
            elif kept_source:
                ok, reason = False, "translate_kept_source"
            else:
                # rewrote but not to official target
                ok, reason = False, "translate_wrong_target"

        rows.append(
            {
                "taxonomy": r["taxonomy"],
                "span": src[:200],
                "expect": expect,
                "pred": pred,
                "output": (out[:160] if out else None),
                "matched": True,
                "missing": False,
                "ok": ok,
                "reason": reason,
                "matched_gt_tgt": matched_tgt,
                "kept_source": kept_source,
                "metric": "act",
            }
        )
    return rows


# Back-compat aliases used by older callers
def match_decision(inv: list[dict], preds: list[dict], check_output: bool = False) -> list[dict]:
    if check_output:
        return score_act(inv, preds)
    return score_decision_probe(inv, preds)


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
