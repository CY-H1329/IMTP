#!/usr/bin/env python3
"""Pack news eval + crops + runner for an external H100 box (Qwen3.8-27B / Gemma-3-27B)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ICLR/
SB = ROOT / "selective_bench"
PACK = ROOT / "h100_news_pack"
GOLD = SB / "gold" / "news_eval.json"

SCRIPTS = [
    "run_news_tables.py",
    "score_news_tables.py",
    "oracle_guide.py",
    "run_rule_yesno.py",
    "run_probe_chain.py",
]


def crop_key(item: dict) -> str:
    aid = item.get("article_id") or item["id"]
    return aid.replace("/", "_")


def src_image(item: dict) -> Path:
    p = Path(item["image"])
    for c in (
        p,
        Path(str(p).replace("/workspace/", "/root/Desktop/workspace/")),
        Path(str(p).replace("/root/Desktop/workspace/", "/workspace/")),
        ROOT / p,
    ):
        if c.exists():
            return c
    return p


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    items = gold["items"]
    PACK.mkdir(parents=True, exist_ok=True)
    crops = PACK / "crops"
    crops.mkdir(exist_ok=True)

    packed = []
    missing = 0
    for it in items:
        src = src_image(it)
        lang = Path(it["image"]).stem
        rel = f"crops/{crop_key(it)}/{lang}.png"
        dst = PACK / rel
        if src.exists():
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                copy_file(src, dst)
        else:
            missing += 1
            print("MISSING", src)
        nit = dict(it)
        nit["image"] = rel
        packed.append(nit)

    out_gold = dict(gold)
    out_gold["items"] = packed
    out_gold["image_root"] = "crops/"
    gold_txt = json.dumps(out_gold, ensure_ascii=False, indent=2)
    (PACK / "gold").mkdir(exist_ok=True)
    (PACK / "gold" / "news_eval.json").write_text(gold_txt, encoding="utf-8")
    sb_gold = PACK / "selective_bench" / "gold"
    sb_gold.mkdir(parents=True, exist_ok=True)
    (sb_gold / "news_eval.json").write_text(gold_txt, encoding="utf-8")

    for name in SCRIPTS:
        copy_file(SB / "scripts" / name, PACK / "selective_bench" / "scripts" / name)
    copy_file(SB / "gold" / "rules.json", PACK / "selective_bench" / "gold" / "rules.json")
    copy_file(
        SB / "gold" / "probe_chain_protocol.json",
        PACK / "selective_bench" / "gold" / "probe_chain_protocol.json",
    )
    copy_file(SB / "prompts" / "rule_yesno.txt", PACK / "selective_bench" / "prompts" / "rule_yesno.txt")
    copy_file(ROOT / "eval_decision_local.py", PACK / "eval_decision_local.py")
    copy_file(ROOT / "H100_RUN.md", PACK / "H100_RUN.md")
    copy_file(ROOT / "H100_RUN.md", SB / "scripts" / "H100_RUN.md")
    copy_file(SB / "scripts" / "run_qwen38_h100.sh", PACK / "run_qwen38_h100.sh")
    (PACK / "run_qwen38_h100.sh").chmod(0o755)

    req = PACK / "requirements.txt"
    req.write_text(
        "\n".join(
            [
                "torch>=2.6",
                "transformers>=4.56",
                "accelerate",
                "pillow",
                "qwen-vl-utils",
                "sentencepiece",
                "protobuf",
                "kernels",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"pack={PACK} n={len(packed)} missing_src={missing}")
    tar = ROOT / "h100_news_pack.tar"
    zst = ROOT / "h100_news_pack.tar.zst"
    overlay = ROOT / "h100_qwen38_scripts.tar.gz"
    subprocess.check_call(
        [
            "tar",
            "czf",
            str(overlay),
            "-C",
            str(PACK),
            "H100_RUN.md",
            "requirements.txt",
            "eval_decision_local.py",
            "run_qwen38_h100.sh",
            "selective_bench/scripts",
            "selective_bench/gold/rules.json",
            "selective_bench/gold/probe_chain_protocol.json",
            "selective_bench/prompts",
        ]
    )
    print("wrote", overlay)
    print("rebuild full archive with:")
    print(f"  tar cf {tar} -C {ROOT} h100_news_pack && zstd -f -19 {tar} -o {zst}")


if __name__ == "__main__":
    main()
