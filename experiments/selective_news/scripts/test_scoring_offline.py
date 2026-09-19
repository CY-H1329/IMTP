#!/usr/bin/env python3
"""Offline tests: Decision probe vs ACT scoring (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strict_score import score_act, score_decision_probe

INV = [
    {
        "text": "Hello World Headline",
        "decision": "translate",
        "taxonomy": "HEADLINE",
        "gt_tgt": "안녕하세요 헤드라인",
    },
    {
        "text": "STORE SIGN",
        "decision": "preserve",
        "taxonomy": "PHOTO_SIGN",
        "gt_tgt": "STORE SIGN",
    },
]


def main() -> None:
    # --- Decision probe ---
    dec_ok = score_decision_probe(
        INV,
        [
            {"text": "Hello World Headline", "decision": "translate"},
            {"text": "STORE SIGN", "decision": "preserve"},
        ],
    )
    assert all(r["ok"] for r in dec_ok), dec_ok

    dec_bad = score_decision_probe(
        INV,
        [
            {"text": "Hello World Headline", "decision": "preserve"},  # wrong
            {"text": "STORE SIGN", "decision": "preserve"},
        ],
    )
    assert dec_bad[0]["ok"] is False and dec_bad[1]["ok"] is True

    # --- Soft artefact that used to inflate unguided ---
    # No JSON spans at all → must FAIL (not OK via kept_source=False)
    empty = score_act(INV, [], raw="random UI chrome A+ A- share button")
    assert all(r["missing"] and not r["ok"] for r in empty), empty

    # Unguided keeps EN source as "translation" → FAIL
    keep_src = score_act(
        INV,
        [
            {
                "text": "Hello World Headline",
                "decision": "translate",
                "output": "Hello World Headline",
            },
            {"text": "STORE SIGN", "decision": "preserve", "output": "STORE SIGN"},
        ],
    )
    assert keep_src[0]["ok"] is False and keep_src[0]["reason"] == "translate_kept_source"
    assert keep_src[1]["ok"] is True

    # Guided does real translation + preserve → OK
    guided = score_act(
        INV,
        [
            {
                "text": "Hello World Headline",
                "decision": "translate",
                "output": "안녕하세요 헤드라인",
            },
            {"text": "STORE SIGN", "decision": "preserve", "output": "STORE SIGN"},
        ],
    )
    assert guided[0]["ok"] and guided[0]["matched_gt_tgt"]
    assert guided[1]["ok"]

    # Wrong target (not gt) → FAIL
    wrong = score_act(
        INV,
        [
            {
                "text": "Hello World Headline",
                "decision": "translate",
                "output": "완전 다른 번역",
            },
            {"text": "STORE SIGN", "decision": "preserve", "output": "STORE SIGN"},
        ],
    )
    assert wrong[0]["ok"] is False and wrong[0]["reason"] == "translate_wrong_target"

    print("OK — Decision probe + ACT scoring behave as intended")
    print("  empty unguided spans → fail (no soft free pass)")
    print("  translate kept source → fail")
    print("  translate ≈ gt_tgt → ok")
    print("  preserve ≈ source → ok")


if __name__ == "__main__":
    main()
