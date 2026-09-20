#!/usr/bin/env python3
"""Isolated-session probe chain on sample_500 (ads/video/news × 500).

Each probe is a brand-new session: one user message, no history, one question.
The experimenter gates later stages on earlier answers; the model never sees them.

S0_present  image: is this element in the image? YES/NO only
S0_quote    image: quote that element, or NONE (asked only if present=YES)
S1          text: is this span that class? YES/NO only
S2          text: TRANSLATE or PRESERVE
S3          image: TRANSLATE or PRESERVE for that span
S4          image: generate localization; score whether the span was actually translated
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import os

def _sb() -> Path:
    e = os.environ.get("SELECTIVE_BENCH_ROOT")
    if e:
        return Path(e)
    here = Path(__file__).resolve().parent.parent
    if (here / "gold").exists():
        return here
    return Path("/workspace/chanyeong/ICLR/selective_bench")

ROOT = _sb()
os.environ.setdefault("SELECTIVE_BENCH_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from oracle_guide import inventory as gold_inventory, score_gen as score_gold_gen  # noqa: E402
from run_rule_yesno import ALIASES, LocalVLM, parse_yn  # noqa: E402

SAMPLE = ROOT / "gold" / "sample_500.json"
PROTO = json.loads((ROOT / "gold" / "probe_chain_protocol.json").read_text())
DEFAULT_DEST = ROOT / "results" / "probe_chain"
PROMPTS = PROTO["prompts"]

TGT_NAME = {
    "en": "English",
    "fr": "French",
    "ko": "Korean",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
}


def parse_tp(text: str):
    t = re.sub(r"[^A-Z]+", " ", (text or "").strip().upper())
    for tok in t.split():
        if tok in ("TRANSLATE", "PRESERVE"):
            return tok.lower()
    if "TRANSLATE" in t:
        return "translate"
    if "PRESERVE" in t:
        return "preserve"
    return None


def parse_quote(raw: str) -> str:
    t = (raw or "").strip()
    m = re.search(r"SPAN:\s*(.+)", t, re.I | re.S)
    if m:
        t = m.group(1).splitlines()[0].strip()
    t = t.strip().strip("\"'`").strip()
    first = t.splitlines()[0].strip().strip("\"'`").strip() if t else ""
    if first.upper() in ("NONE", "N/A", "NO", "-", "NULL", "N.A."):
        return ""
    return first


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


TAX_ALIASES = {
    "C3": {"C3", "OFFICIAL_KEEP"},
    "OFFICIAL_KEEP": {"C3", "OFFICIAL_KEEP"},
}


def gold_span(item: dict, taxonomy: str) -> dict | None:
    want = TAX_ALIASES.get(taxonomy, {taxonomy})
    best = None
    for s in item.get("spans") or []:
        if s.get("taxonomy") not in want or not s.get("text"):
            continue
        if best is None or len(s["text"]) > len(best["text"]):
            best = s
    return best


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict) and "id" in rec:
                done.add(rec["id"])
    return done


def ask(vlm: LocalVLM, image: Path | None, prompt: str, max_new: int | None = None) -> str:
    """One isolated generate: no chat history, no prior answers in the prompt."""
    if image is None:
        blank = ROOT / "results" / "_blank.png"
        return vlm.generate(blank, prompt, max_new=max_new)
    return vlm.generate(image, prompt, max_new=max_new)


def run_item(vlm: LocalVLM, item: dict) -> dict:
    cat = item["category"]
    rules = PROTO["rules"][cat]
    present = set()
    for s in item.get("spans") or []:
        tax = s.get("taxonomy")
        if not tax:
            continue
        present.add(tax)
        present.update(TAX_ALIASES.get(tax, set()))
    img = Path(item["image"])
    tgt = item.get("tgt_lang") or "fr"
    tgt_n = TGT_NAME.get(tgt, tgt)
    s0 = []
    found: list[tuple[dict, str, dict | None]] = []

    for rule in rules:
        tax = rule["taxonomy"]
        gold = gold_span(item, tax)
        expect = "YES" if (tax in present or gold) else "NO"
        raw_p = ask(vlm, img, PROMPTS["S0_present"].format(element=rule["element"]), max_new=8)
        yn = parse_yn(raw_p)
        span = ""
        raw_q = ""
        if yn == "YES":
            raw_q = ask(vlm, img, PROMPTS["S0_quote"].format(element=rule["element"]), max_new=64)
            span = parse_quote(raw_q)
        id_ok = None
        if yn == "YES" and span and gold:
            id_ok = similar(span, gold["text"])
        elif yn == "NO":
            id_ok = expect == "NO"
        rec = {
            "stage": "S0",
            "rule_id": rule["id"],
            "taxonomy": tax,
            "expect_present": expect,
            "pred_present": yn,
            "present_ok": yn == expect,
            "ok": yn == expect,
            "span": span[:200],
            "gold_span": (gold or {}).get("text", "")[:200],
            "identify_ok": id_ok,
            "raw_present": (raw_p or "")[:400],
            "raw_quote": (raw_q or "")[:400],
        }
        s0.append(rec)
        if yn == "YES" and span:
            found.append((rule, span, gold))

    span_probes = []
    for rule, span, gold in found:
        gold_d = ((gold or {}).get("decision") or rule.get("gold_decision") or "").lower()
        raw = ask(
            vlm,
            None,
            PROMPTS["S1_classify"].format(element=rule["element"], span=span),
            max_new=8,
        )
        yn = parse_yn(raw)
        span_probes.append(
            {
                "stage": "S1_classify",
                "rule_id": rule["id"],
                "taxonomy": rule["taxonomy"],
                "span": span[:200],
                "expect": "YES",
                "pred": yn,
                "ok": yn == "YES",
                "raw": (raw or "")[:240],
            }
        )
        raw = ask(vlm, None, PROMPTS["S2_policy"].format(tgt_n=tgt_n, span=span), max_new=8)
        tp = parse_tp(raw)
        span_probes.append(
            {
                "stage": "S2_policy",
                "rule_id": rule["id"],
                "taxonomy": rule["taxonomy"],
                "span": span[:200],
                "expect": gold_d,
                "pred": tp,
                "ok": tp == gold_d if gold_d else None,
                "raw": (raw or "")[:240],
            }
        )
        raw = ask(vlm, img, PROMPTS["S3_decide"].format(tgt_n=tgt_n, span=span), max_new=8)
        tp = parse_tp(raw)
        span_probes.append(
            {
                "stage": "S3_decide_image",
                "rule_id": rule["id"],
                "taxonomy": rule["taxonomy"],
                "span": span[:200],
                "expect": gold_d,
                "pred": tp,
                "ok": tp == gold_d if gold_d else None,
                "raw": (raw or "")[:240],
            }
        )

    gen = ask(vlm, img, PROMPTS["S4_generate"].format(tgt_n=tgt_n), max_new=400)
    behavior = []
    for rule, span, gold in found:
        gold_d = ((gold or {}).get("decision") or rule.get("gold_decision") or "").lower()
        tgt_txt = (gold or {}).get("gt_tgt") or ""
        did_keep = similar(span, gen)
        did_tgt = similar(tgt_txt, gen) if tgt_txt else False
        if gold_d == "preserve":
            ok = did_keep
        elif gold_d == "translate":
            ok = (not did_keep) or did_tgt
        else:
            ok = None
        behavior.append(
            {
                "stage": "S4_generate",
                "rule_id": rule["id"],
                "taxonomy": rule["taxonomy"],
                "span": span[:200],
                "expect": gold_d,
                "kept_source": did_keep,
                "matched_gt_tgt": did_tgt,
                "ok": ok,
            }
        )

    gold_inv = gold_inventory(item)
    return {
        "id": item["id"],
        "category": cat,
        "image": item["image"],
        "src_lang": item.get("src_lang"),
        "tgt_lang": tgt,
        "S0": s0,
        "span_probes": span_probes,
        "S4": behavior,
        "S4_gold": score_gold_gen(gold_inv, gen or ""),
        "S4_raw": (gen or "")[:4000],
        "condition": "unguided",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3vl"])
    ap.add_argument("--categories", nargs="+", default=["ads", "video", "news"])
    ap.add_argument("--limit", type=int, default=0, help="first N items per category")
    ap.add_argument("--dest-dir", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()
    sample = json.loads(SAMPLE.read_text())
    dest_dir: Path = args.dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    for spec in args.models:
        mid = ALIASES.get(spec, spec)
        slug = mid.split("/")[-1].replace(" ", "_")
        outp = dest_dir / f"{slug}.jsonl"
        done = load_done(outp)
        vlm = LocalVLM(mid, max_side=768, max_new=400)
        n = 0
        with outp.open("a", encoding="utf-8") as f:
            for cat in args.categories:
                items = sample["items"][cat]
                if args.limit:
                    items = items[: args.limit]
                for it in items:
                    if it["id"] in done:
                        continue
                    rec = run_item(vlm, it)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    s0_ok = sum(1 for x in rec["S0"] if x.get("present_ok"))
                    n_found = len(rec["S4"])
                    print(
                        f"{slug} {cat} {it['id'][:40]} "
                        f"S0 {s0_ok}/{len(rec['S0'])} chained={n_found}",
                        flush=True,
                    )
        vlm.close()
        print("wrote", outp, "new", n)


if __name__ == "__main__":
    main()
