"""Re-OCR and lightweight metric helpers."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple


def _parse_paddle_ocr_result(result: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Support PaddleOCR 2.x list format and 3.x predict() dict/OCRResult format."""
    regions: List[Dict[str, Any]] = []
    texts: List[str] = []

    if result is None:
        return regions, texts

    # PaddleOCR 3.x: list of dict-like results with rec_texts / dt_polys
    if isinstance(result, list) and result and isinstance(result[0], dict):
        for item in result:
            rec_texts = item.get("rec_texts") or item.get("rec_text") or []
            rec_scores = item.get("rec_scores") or item.get("rec_score") or []
            polys = item.get("dt_polys") or item.get("rec_polys") or item.get("dt_boxes") or []
            if isinstance(rec_texts, str):
                rec_texts = [rec_texts]
            if not isinstance(rec_scores, (list, tuple)):
                rec_scores = [rec_scores] * len(rec_texts)
            for i, txt in enumerate(rec_texts):
                conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                bbox = [0, 0, 0, 0]
                if i < len(polys) and polys[i] is not None:
                    poly = polys[i]
                    try:
                        xs = [float(p[0]) for p in poly]
                        ys = [float(p[1]) for p in poly]
                        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                    except Exception:
                        pass
                regions.append({"bbox": bbox, "text": str(txt), "confidence": conf})
                texts.append(str(txt))
        return regions, texts

    # PaddleOCR 3.x OCRResult objects (attribute access)
    if isinstance(result, list) and result and hasattr(result[0], "get"):
        try:
            return _parse_paddle_ocr_result([dict(r) if not isinstance(r, dict) else r for r in result])
        except Exception:
            pass
    if isinstance(result, list) and result and hasattr(result[0], "rec_texts"):
        converted = []
        for r in result:
            converted.append({
                "rec_texts": getattr(r, "rec_texts", []),
                "rec_scores": getattr(r, "rec_scores", []),
                "dt_polys": getattr(r, "dt_polys", getattr(r, "rec_polys", [])),
            })
        return _parse_paddle_ocr_result(converted)

    # PaddleOCR 2.x: [[[box], (text, conf)], ...]
    for block in result or []:
        if not block:
            continue
        # Sometimes a single page is already a list of lines
        lines = block if isinstance(block[0], (list, tuple)) and len(block[0]) == 2 else [block]
        for line in lines:
            try:
                box, meta = line
                if isinstance(meta, (list, tuple)):
                    txt, conf = meta[0], float(meta[1])
                else:
                    txt, conf = str(meta), 0.0
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                regions.append({"bbox": bbox, "text": txt, "confidence": conf})
                texts.append(txt)
            except Exception:
                continue
    return regions, texts


def _configure_paddle_runtime() -> None:
    """Disable OneDNN/MKLDNN paths that crash on some GPU servers (PIR ArrayAttribute)."""
    import os

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_onednn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        import paddle

        paddle.set_flags({
            "FLAGS_use_mkldnn": False,
            "FLAGS_enable_pir_in_executor": False,
        })
    except Exception:
        pass


def _make_paddle_ocr():
    from paddleocr import PaddleOCR

    # Prefer GPU when visible; fall back to CPU without MKLDNN.
    device_candidates = []
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            device_candidates.append("gpu")
            device_candidates.append("gpu:0")
    except Exception:
        pass
    device_candidates.append("cpu")

    last_err: Optional[Exception] = None
    for device in device_candidates:
        kwargs_list = [
            {"lang": "en", "device": device, "enable_mkldnn": False},
            {"lang": "en", "device": device},
            {"lang": "en", "enable_mkldnn": False},
            {"lang": "en"},
            {"use_angle_cls": True, "lang": "en"},
        ]
        for kwargs in kwargs_list:
            try:
                return PaddleOCR(**kwargs)
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
    if last_err:
        raise last_err
    raise RuntimeError("Failed to construct PaddleOCR")


def run_paddle_ocr(image_path: str) -> Dict[str, Any]:
    try:
        _configure_paddle_runtime()
        from paddleocr import PaddleOCR  # noqa: F401 — import check
    except ImportError as e:
        return {"status": "skipped", "error": str(e), "regions": [], "full_text": ""}

    try:
        ocr = _make_paddle_ocr()

        if hasattr(ocr, "predict"):
            raw = ocr.predict(image_path)
        else:
            try:
                raw = ocr.ocr(image_path, cls=True)
            except TypeError:
                raw = ocr.ocr(image_path)

        regions, texts = _parse_paddle_ocr_result(raw)
        return {
            "status": "ok",
            "engine": "paddleocr",
            "regions": regions,
            "full_text": " ".join(texts),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "regions": [], "full_text": ""}


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
