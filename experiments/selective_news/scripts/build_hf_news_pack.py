#!/usr/bin/env python3
"""Build HuggingFace-ready news pack: one record per article.

- Dong-A: up to 4 crops (en, ko, ja/jp, zh)
- Chosun: 2 crops (en, ko), PHOTO_SIGN-only (from news_eval)
- GT spans / decisions copied as-is (no logo/masthead GT edits)
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path("/root/Desktop/workspace/chanyeong/ICLR")
GOLD_EVAL = ROOT / "selective_bench" / "gold" / "news_eval.json"
GOLD_FULL = ROOT / "selective_bench" / "gold" / "news_3000.json"
OUT = ROOT / "hf_news_pack"
README = OUT / "README.md"


def fix(p: str) -> Path:
    return Path(str(p).replace("/workspace/", "/root/Desktop/workspace/"))


def crop_langs(article_id: str, source: str, sample_image: str) -> dict[str, Path]:
    """Discover available language crops next to the eval image."""
    img = fix(sample_image)
    crop_dir = img.parent
    out: dict[str, Path] = {}
    if not crop_dir.exists():
        return out
    for p in sorted(crop_dir.glob("*.png")):
        lang = p.stem.lower()
        if lang == "jp":
            lang = "ja"
        out[lang] = p
    # expected sets
    if source == "donga":
        prefer = ["en", "ko", "ja", "zh"]
    else:
        prefer = ["en", "ko"]
    return {k: out[k] for k in prefer if k in out} or out


def main() -> None:
    eval_gold = json.loads(GOLD_EVAL.read_text(encoding="utf-8"))
    full = json.loads(GOLD_FULL.read_text(encoding="utf-8"))
    pairs_by = defaultdict(list)
    for it in full["items"]:
        pairs_by[it["article_id"]].append(
            {
                "id": it["id"],
                "src_lang": it["src_lang"],
                "tgt_lang": it["tgt_lang"],
                "url_src": it.get("url_src"),
                "url_tgt": it.get("url_tgt"),
                "spans": it["spans"],  # GT as-is
            }
        )

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "crops").mkdir(parents=True)

    articles = []
    missing_crop = 0
    n_donga = n_chosun = 0
    for it in eval_gold["items"]:
        aid = it["article_id"]
        source = it["source"]
        langs = crop_langs(aid, source, it["image"])
        if not langs:
            missing_crop += 1
            continue
        # copy crops → crops/<article_id>/<lang>.png
        rel_images = {}
        for lang, src in langs.items():
            dst = OUT / "crops" / aid / f"{lang}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            rel_images[lang] = f"crops/{aid}/{lang}.png"

        # primary eval pair from news_eval (usually en→ko), GT unchanged
        primary = {
            "id": it["id"],
            "src_lang": it["src_lang"],
            "tgt_lang": it["tgt_lang"],
            "url_src": it.get("url_src"),
            "url_tgt": it.get("url_tgt"),
            "image": rel_images.get(it["src_lang"]) or next(iter(rel_images.values())),
            "spans": it["spans"],
        }
        rec = {
            "article_id": aid,
            "source": source,
            "en_id": it.get("en_id"),
            "n_images": len(rel_images),
            "images": rel_images,  # donga: ≤4, chosun: ≤2
            "eval_pair": primary,
            "all_pairs": pairs_by.get(aid, [primary]),
            "criterion_note": (
                "PHOTO_SIGN=preserve (diegetic on-photo text). "
                "HEADLINE/BODY/MASTHEAD=translate per official editions. "
                "GT labels are frozen — do not rewrite logos/mastheads by hand."
            ),
        }
        articles.append(rec)
        if source == "donga":
            n_donga += 1
        else:
            n_chosun += 1

    meta = {
        "task": "selective_translation_news",
        "n_articles": len(articles),
        "n_donga": n_donga,
        "n_chosun": n_chosun,
        "layout": {
            "donga": "4 official editions → crops en/ko/ja/zh when present",
            "chosun": "2 editions → crops en/ko; PHOTO_SIGN-only subset",
        },
        "eval": "Use eval_pair (one src→tgt per article, prefer en→ko). all_pairs optional.",
        "gt_policy": "Do not alter logo/masthead/PHOTO_SIGN gold labels when exporting.",
        "missing_crop_articles": missing_crop,
        "articles": articles,
    }
    (OUT / "articles.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # also flat jsonl for datasets.load_dataset
    with (OUT / "articles.jsonl").open("w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    README.write_text(
        f"""---
license: other
task_categories:
  - image-text-to-text
language:
  - en
  - ko
  - ja
  - zh
pretty_name: Selective News (Dong-A / Chosun)
size_categories:
  - 1K<n<10K
---

# Selective translation — news (private)

**n_articles = {len(articles)}** (Dong-A {n_donga}, Chosun {n_chosun}).

## Layout

| Source | Images / article | Languages |
|---|---|---|
| Dong-A | up to **4** | `en`, `ko`, `ja`, `zh` |
| Chosun | **2** | `en`, `ko` (PHOTO_SIGN / on-photo text only) |

Each article record:

- `images`: map lang → `crops/<article_id>/<lang>.png`
- `eval_pair`: one src→tgt for benchmark (prefer en→ko), with gold `spans`
- `all_pairs`: all official parallel pairs from `news_3000` (optional)

## Gold spans (frozen)

Per eval pair, usually 4 spans:

| taxonomy | decision |
|---|---|
| HEADLINE | translate |
| BODY | translate |
| MASTHEAD | translate (official edition form — **do not invent**; keep GT as labeled) |
| PHOTO_SIGN | **preserve** (diegetic text inside the photo) |

Do **not** change logo / masthead / on-photo GT when editing the pack.

## Files

```
articles.json      # full dump + metadata
articles.jsonl     # one article per line
crops/...          # language crops
README.md
```

## Load

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="articles.jsonl", split="train")
```

## Privacy

Newspaper crops + official text → keep **private** / research-gated.
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT} n={len(articles)} donga={n_donga} chosun={n_chosun} missing={missing_crop}")
    # verify image counts
    from collections import Counter
    c = Counter(a["n_images"] for a in articles if a["source"] == "donga")
    k = Counter(a["n_images"] for a in articles if a["source"] == "chosun")
    print("donga n_images dist", dict(c))
    print("chosun n_images dist", dict(k))


if __name__ == "__main__":
    main()
