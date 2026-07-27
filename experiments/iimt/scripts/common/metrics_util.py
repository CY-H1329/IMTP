"""Re-OCR and lightweight metric helpers."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple


def run_paddle_ocr(image_path: str) -> Dict[str, Any]:
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        return {"status": "skipped", "error": str(e), "regions": [], "full_text": ""}

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr.ocr(image_path, cls=True)
    regions: List[Dict[str, Any]] = []
    texts: List[str] = []
    for block in result or []:
        for line in block or []:
            box, (txt, conf) = line
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            regions.append({"bbox": bbox, "text": txt, "confidence": float(conf)})
            texts.append(txt)
    return {
        "status": "ok",
        "engine": "paddleocr",
        "regions": regions,
        "full_text": " ".join(texts),
    }


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref or "", hyp or ""
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        import jiwer
        return float(jiwer.cer(ref, hyp))
    except Exception:
        # fallback char-level edit distance
        return _levenshtein_chars(ref, hyp) / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    ref, hyp = ref or "", hyp or ""
    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0
    try:
        import jiwer
        return float(jiwer.wer(ref, hyp))
    except Exception:
        return _levenshtein_words(ref, hyp) / max(len(ref.split()), 1)


def exact_match(ref: str, hyp: str) -> int:
    return int((ref or "").strip() == (hyp or "").strip())


def _levenshtein_chars(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _levenshtein_words(a: str, b: str) -> int:
    wa, wb = a.split(), b.split()
    if not wa:
        return len(wb)
    if not wb:
        return len(wa)
    prev = list(range(len(wb) + 1))
    for i, wa_i in enumerate(wa, 1):
        cur = [i]
        for j, wb_j in enumerate(wb, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (wa_i != wb_j)))
        prev = cur
    return prev[-1]


def bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
