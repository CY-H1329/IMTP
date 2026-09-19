"""Oracle guide for S4: rules + proper-noun identity + exact TRANSLATE/PRESERVE.

Does not leak gold target strings. Shared by guided generation and unguided gold scoring.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import os

def _sb() -> Path:
    e = os.environ.get("SELECTIVE_BENCH_ROOT")
    if e:
        return Path(e)
    here = Path(__file__).resolve().parent
    cand = here.parent
    if (cand / "gold" / "rules.json").exists():
        return cand
    return Path("/workspace/chanyeong/ICLR/selective_bench")

ROOT = _sb()
RULES = json.loads((ROOT / "gold" / "rules.json").read_text())

KIND = {
    "C1": {"role": "brand name", "proper_noun": True},
    "C2": {"role": "product name", "proper_noun": True},
    "COPY": {"role": "marketing copy (headline / slogan / CTA / claims)", "proper_noun": False},
    "C3": {"role": "text printed on the physical pack", "proper_noun": False},
    "B3": {"role": "ingredient list / model code / net weight / SPF / unit on the pack", "proper_noun": False},
    "CAPTION": {"role": "burned-in subtitle / caption (timed-text layer)", "proper_noun": False},
    "OFFICIAL_KEEP": {"role": "in-world video text (logo / sign / watermark / UI)", "proper_noun": False},
    "HEADLINE": {"role": "news headline", "proper_noun": False},
    "BODY": {"role": "article body / standfirst", "proper_noun": False},
    "MASTHEAD": {"role": "newspaper masthead / wordmark", "proper_noun": True},
    "PHOTO_SIGN": {"role": "text inside the photograph (building sign / plaque / diegetic lettering)", "proper_noun": True},
    "PERSON": {"role": "person name", "proper_noun": True},
    "ORG": {"role": "organization name", "proper_noun": True},
}

TGT_NAME = {
    "en": "English",
    "fr": "French",
    "ko": "Korean",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
}


def norm(s: str) -> str:
    s = str(s or "").lower().replace("\n", " ")
    s = re.sub(r"[^\w가-힣ぁ-んァ-ン一-龥]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def similar(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.55


def kind_of(taxonomy: str) -> dict:
    return KIND.get(taxonomy or "", {"role": taxonomy or "text", "proper_noun": False})


def inventory(item: dict) -> list[dict]:
    rows = []
    seen = set()
    for s in item.get("spans") or []:
        text = (s.get("text") or "").strip()
        d = (s.get("decision") or "").lower()
        if not text or d not in ("translate", "preserve"):
            continue
        key = (norm(text), d, s.get("taxonomy") or "")
        if key in seen:
            continue
        seen.add(key)
        k = kind_of(s.get("taxonomy") or "")
        rows.append(
            {
                "text": text,
                "taxonomy": s.get("taxonomy") or "",
                "role": k["role"],
                "proper_noun": bool(k["proper_noun"]),
                "decision": d,
                "gt_tgt": s.get("gt_tgt") or "",
            }
        )
    return rows


def rules_block(category: str) -> str:
    lines = []
    for r in RULES.get(category) or []:
        if r.get("dropped"):
            continue
        rid = r.get("id") or ""
        rule = (r.get("rule") or "").strip()
        if not rule:
            continue
        gold = (r.get("gold") or "").upper()
        lines.append(f"- [{rid}] {rule} ({gold})")
    if category == "news":
        lines.append(
            "- [NP/NO] Person names and organization names are proper nouns: "
            "use the official target-language form (translate if the official edition does)."
        )
    return "\n".join(lines) if lines else "- (no rules)"


def guided_prompt(item: dict, inv: list[dict] | None = None) -> str:
    inv = inventory(item) if inv is None else inv
    tgt = item.get("tgt_lang") or "fr"
    tgt_n = TGT_NAME.get(tgt, tgt)
    cat = item.get("category") or "ads"
    proper = [r for r in inv if r["proper_noun"]]
    translate = [r for r in inv if r["decision"] == "translate"]
    preserve = [r for r in inv if r["decision"] == "preserve"]

    def line(r: dict) -> str:
        pn = " [proper noun]" if r["proper_noun"] else ""
        return f'- [{r["role"]}]{pn} "{r["text"]}"'

    proper_blk = "\n".join(line(r) for r in proper) if proper else "- (none in the gold inventory)"
    tr_blk = "\n".join(line(r) for r in translate) if translate else "- (none)"
    pr_blk = "\n".join(line(r) for r in preserve) if preserve else "- (none)"
    return (
        f"Localize this image into {tgt_n}.\n\n"
        "Follow the constructional rules and the exact inventory below. "
        "Do not invent extra spans. Do not ignore a listed span.\n\n"
        f"RULES:\n{rules_block(cat)}\n\n"
        "PROPER NOUNS in this image (identity). "
        "A proper noun is a brand, product, person, organization, newspaper name, "
        "or diegetic sign/lettering inside a photograph:\n"
        f"{proper_blk}\n\n"
        f"TRANSLATE into {tgt_n} (must not stay in the source language):\n"
        f"{tr_blk}\n\n"
        "PRESERVE unchanged (copy the source string exactly; do not translate):\n"
        f"{pr_blk}\n\n"
        "Output a JSON list of objects "
        '{"text":"...","decision":"translate"|"preserve","output":"..."}.\n'
        "For PRESERVE, output must equal the source text. "
        "For TRANSLATE, output must be in "
        f"{tgt_n}. No extra commentary."
    )


def score_gen(inv: list[dict], gen: str) -> list[dict]:
    out = []
    for r in inv:
        gold_d = r["decision"]
        did_keep = similar(r["text"], gen)
        did_tgt = similar(r.get("gt_tgt") or "", gen) if r.get("gt_tgt") else False
        if gold_d == "preserve":
            ok = did_keep
        else:
            ok = (not did_keep) or did_tgt
        out.append(
            {
                "taxonomy": r["taxonomy"],
                "role": r["role"],
                "proper_noun": r["proper_noun"],
                "span": r["text"][:200],
                "expect": gold_d,
                "kept_source": did_keep,
                "matched_gt_tgt": did_tgt,
                "ok": ok,
            }
        )
    return out
