#!/usr/bin/env python3
"""Portable root + image resolution for selective_news (local or H100)."""
from __future__ import annotations

import os
import re
from pathlib import Path


def sb_root() -> Path:
    e = os.environ.get("SELECTIVE_BENCH_ROOT")
    if e:
        return Path(e)
    here = Path(__file__).resolve().parent.parent
    if (here / "gold").exists():
        return here
    for p in (
        Path("/workspace/chanyeong/ICLR/selective_bench"),
        Path("/root/Desktop/workspace/chanyeong/ICLR/selective_bench"),
    ):
        if (p / "gold").exists():
            return p
    return here


def news_data_root() -> Path | None:
    e = os.environ.get("NEWS_DATA")
    if e and Path(e).exists():
        return Path(e)
    root = sb_root()
    for cand in (root / "data" / "hf_news_pack", root / "data"):
        if cand.exists():
            return cand
    return None


_CORPUS_RE = re.compile(
    r"(?:/|^)(?:news/)?(?P<source>donga|chosun)_corpus/(?P<eid>[^/]+)/crop/(?P<lang>[^.]+)\.png$",
    re.I,
)
_PACK_RE = re.compile(
    r"crops/(?P<aid>(?:donga|chosun)_[^/]+)/(?P<lang>[^.]+)\.png$",
    re.I,
)


def _remap_workspace(p: str) -> list[Path]:
    s = str(p)
    out = [Path(s)]
    out.append(Path(s.replace("/workspace/", "/root/Desktop/workspace/")))
    out.append(Path(s.replace("/root/Desktop/workspace/", "/workspace/")))
    return out


def resolve_image(p: str, article_id: str | None = None, src_lang: str | None = None) -> Path:
    """Resolve gold image path against NEWS_DATA / hf_news_pack / local corpus."""
    raw = str(p)
    for c in _remap_workspace(raw):
        if c.exists():
            return c

    nd = news_data_root()
    lang = (src_lang or Path(raw).stem).lower()
    if lang == "jp":
        lang = "ja"

    # Already pack-relative
    m = _PACK_RE.search(raw.replace("\\", "/"))
    if m and nd:
        cand = nd / "crops" / m.group("aid") / f"{m.group('lang').lower()}.png"
        if cand.exists():
            return cand

    # Absolute corpus → pack layout crops/<source>_<eid>/<lang>.png
    m = _CORPUS_RE.search(raw.replace("\\", "/"))
    if m:
        source, eid, clang = m.group("source").lower(), m.group("eid"), m.group("lang").lower()
        if clang == "jp":
            clang = "ja"
        aid = article_id or f"{source}_{eid}"
        if nd:
            for base in (nd, nd / "hf_news_pack" if (nd / "hf_news_pack").exists() else nd):
                cand = base / "crops" / aid / f"{clang}.png"
                if cand.exists():
                    return cand
                # also try eid-only folder names used in some snapshots
                cand2 = base / "crops" / source / eid / f"{clang}.png"
                if cand2.exists():
                    return cand2

    if article_id and nd:
        cand = nd / "crops" / article_id / f"{lang}.png"
        if cand.exists():
            return cand

    return Path(raw)
