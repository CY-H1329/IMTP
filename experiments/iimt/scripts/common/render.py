"""Simple text overlay renderer for cascade baselines."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_translations(
    image_path: Path,
    regions: List[Dict[str, Any]],
    out_path: Path,
    *,
    fill: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Path:
    """Paint translated strings into OCR bounding boxes."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for reg in regions:
        bbox = reg.get("bbox")
        text = str(reg.get("translated") or reg.get("text") or "").strip()
        if not bbox or not text:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        draw.rectangle([x1, y1, x2, y2], fill=fill)
        h = max(y2 - y1, 12)
        font = _font(max(10, min(h - 4, 28)))
        draw.text((x1 + 2, y1 + 2), text, fill=text_color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def regions_to_pred_text(regions: List[Dict[str, Any]]) -> str:
    parts = [str(r.get("translated") or r.get("text") or "").strip() for r in regions]
    return " ".join(p for p in parts if p)
