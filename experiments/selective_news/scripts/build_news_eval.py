#!/usr/bin/env python3
"""One eval item per news article (prefer en→ko) from news_3000.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/root/Desktop/workspace/chanyeong/ICLR")
SB = ROOT / "selective_bench"
SRC = SB / "gold" / "news_3000.json"
OUT = SB / "gold" / "news_eval.json"

WS = "/workspace/chanyeong/ICLR"
DESK = "/root/Desktop/workspace/chanyeong/ICLR"


def resolve_image(p: str) -> str:
    cand = Path(p)
    if cand.exists():
        return str(cand)
    alt = Path(str(p).replace(WS, DESK).replace(DESK, WS))
    if alt.exists():
        return str(alt)
    return p


def pick_items(items: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(it["article_id"], []).append(it)
    out = []
    for aid, recs in by.items():
        recs = list(recs)
        chosen = None
        for it in recs:
            if it.get("src_lang") == "en" and it.get("tgt_lang") == "ko":
                chosen = it
                break
        if chosen is None:
            for it in recs:
                if it.get("src_lang") == "en":
                    chosen = it
                    break
        if chosen is None:
            chosen = recs[0]
        rec = dict(chosen)
        rec["category"] = "news"
        rec["image"] = resolve_image(rec.get("image") or "")
        rec["n_pairs"] = len(recs)
        out.append(rec)
    return out


def main() -> None:
    gold = json.loads(SRC.read_text())
    items = pick_items(gold["items"])
    payload = {
        "task": "selective_translation",
        "subset": "news_eval",
        "from": "news_3000",
        "n_articles": len(items),
        "n_items": len(items),
        "n_donga": sum(1 for it in items if it.get("source") == "donga"),
        "n_chosun": sum(1 for it in items if it.get("source") == "chosun"),
        "langs": gold.get("langs"),
        "criterion": gold.get("criterion"),
        "note": "One src→tgt pair per article (prefer en→ko). Image is the source-language crop.",
        "items": items,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = sum(1 for it in items if not Path(it["image"]).exists())
    print(
        f"Wrote {OUT} articles={len(items)} donga={payload['n_donga']} "
        f"chosun={payload['n_chosun']} missing_img={missing}"
    )


if __name__ == "__main__":
    main()
