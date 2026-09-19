#!/usr/bin/env python3
"""GPU local-VLM Decision eval (no API). Same protocol as eval_decision.py.

GT: overlay in texts.json['drawn'][src] = translate
    brand/product in _layout.txt = preserve
    pack-printed English marked translate = error (pct_images_pack_en)

Shortcuts:
  --model qwen      Qwen/Qwen2.5-VL-7B-Instruct
  --model internvl  OpenGVLab/InternVL2_5-8B
  --model llava     llava-hf/llava-onevision-qwen2-7b-ov-hf
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
ADS = Path(os.environ.get("DECISION_ADS", ROOT / "ads"))
OUT = Path(os.environ.get("DECISION_OUT", ROOT / "eval" / "decision"))

LANG_NAME = {
    "en": "English",
    "fr": "French",
    "ko": "Korean",
    "es": "Spanish",
    "zh": "Chinese",
}

MODEL_ALIASES = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL2_5-8B",
    "internvl2.5-8b": "OpenGVLab/InternVL2_5-8B",
    "llava": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava-ov": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
}

PROMPT = """You are looking at a product advertisement image.

The user wants this advertisement translated into {tgt_name}.
Do NOT produce the translation yet.

Task: selective judgment.
1. Read every visible text span in the image.
2. For each span, decide:
   - "translate": marketing copy (headline, slogan, CTA, claims, hashtags, step labels on the ad layout)
   - "preserve": brand names, product names, ingredients (INCI), model codes, net weight/SPF/units, and text printed on the physical pack

Return JSON only, no markdown:
{{"items":[{{"text":"...","decision":"translate"|"preserve"}}]}}
"""


def norm(s: str) -> str:
    s = str(s or "").lower().replace("\n", " ")
    s = re.sub(r"[^\w가-힣]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def split_bits(s: str) -> list[str]:
    out = []
    for part in re.split(r"[\n|/]+", str(s or "")):
        t = part.strip(" -–—\t")
        if len(norm(t)) >= 2:
            out.append(t)
    return out


def uniq(xs: list[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        k = norm(x)
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


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


def hits(pred: list[str], gold: list[str]) -> int:
    used = [False] * len(gold)
    n = 0
    for p in pred:
        for i, g in enumerate(gold):
            if not used[i] and similar(p, g):
                used[i] = True
                n += 1
                break
    return n


def gt_for(folder: Path, src_lang: str) -> dict:
    meta = (folder / "_layout.txt").read_text(encoding="utf-8").splitlines()
    layout = meta[0] if meta else ""
    brand = meta[4].strip() if len(meta) > 4 else ""
    product = meta[5].strip() if len(meta) > 5 else ""
    blob = json.loads((folder / "texts.json").read_text(encoding="utf-8"))
    drawn = (blob.get("drawn") or {}).get(src_lang) or []
    overlay = uniq(list(drawn) if isinstance(drawn, list) else [])
    preserve = uniq([brand, product])
    translate, moved = [], []
    for t in overlay:
        if any(similar(t, p) for p in preserve):
            moved.append(t)
        else:
            translate.append(t)
    preserve = uniq(preserve + moved)
    return {
        "layout": layout,
        "brand": brand,
        "product": product,
        "translate": translate,
        "preserve": preserve,
    }


def list_items(cats: list[str]) -> list[dict]:
    items = []
    for cat in cats:
        d = ADS / cat
        if not d.exists():
            continue
        for folder in sorted(p for p in d.iterdir() if p.is_dir()):
            if not (folder / "texts.json").exists():
                continue
            items.append({"cat": cat, "stem": folder.name, "folder": folder})
    return items


def sample_items(items: list[dict], n: int, seed: int) -> list[dict]:
    by = {}
    for it in items:
        by.setdefault(it["cat"], []).append(it)
    rng = random.Random(seed)
    out = []
    cats = list(by)
    per = max(1, n // max(1, len(cats)))
    for cat in cats:
        pool = by[cat][:]
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    return out[:n]


def parse_json_obj(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a : b + 1])
        raise


def looks_latin_pack(s: str) -> bool:
    t = s.lower()
    if re.search(r"[가-힣]", s):
        return False
    if re.search(r"[àâäéèêëïîôùûçœæ]", t):
        return False
    return bool(re.search(r"[a-z]{3,}", t))


def extra_flags(pred_t: list[str], gt: dict) -> dict:
    gold_t, gold_p = gt["translate"], gt["preserve"]
    extra = [p for p in pred_t if not any(similar(p, g) for g in gold_t)]
    pack_en = [p for p in extra if looks_latin_pack(p) and not any(similar(p, g) for g in gold_p)]
    brand_as_t = [p for p in pred_t if any(similar(p, g) for g in gold_p)]
    return {
        "n_extra": len(extra),
        "has_extra": int(len(extra) > 0),
        "n_pack_en": len(pack_en),
        "has_pack_en": int(len(pack_en) > 0),
        "has_brand_t": int(len(brand_as_t) > 0),
    }


def score_one(pred_items: list[dict], gt: dict) -> dict:
    pred_t = [x["text"] for x in pred_items if str(x.get("decision", "")).lower().startswith("t")]
    pred_p = [x["text"] for x in pred_items if str(x.get("decision", "")).lower().startswith("p")]
    gold_t, gold_p = gt["translate"], gt["preserve"]
    tp_t = hits(pred_t, gold_t)
    rec_t = tp_t / max(1, len(gold_t))
    prec_t = tp_t / max(1, len(pred_t))
    over = hits(pred_t, gold_p) / max(1, len(gold_p))
    return {
        "n_pred_translate": len(pred_t),
        "n_pred_preserve": len(pred_p),
        "n_gt_translate": len(gold_t),
        "n_gt_preserve": len(gold_p),
        "translate_recall": rec_t,
        "translate_precision": prec_t,
        "over_translation": over,
        "under_translation": 1.0 - rec_t,
        "pred_translate": pred_t,
        "pred_preserve": pred_p,
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def resize_rgb(path: Path, max_side: int = 768) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def family_of(model_id: str) -> str:
    m = model_id.lower()
    if "internvl" in m:
        return "internvl"
    if "llava" in m or "onevision" in m:
        return "llava"
    return "qwen"


class LocalVLM:
    def __init__(self, model_id: str, max_side: int = 768):
        self.model_id = model_id
        self.family = family_of(model_id)
        self.max_side = max_side
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        import torch

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"loading {self.model_id} family={self.family} cuda={torch.cuda.is_available()}", flush=True)
        if self.family == "internvl":
            self._load_internvl(dtype)
        elif self.family == "llava":
            self._load_llava(dtype)
        else:
            self._load_qwen(dtype)

    def _load_qwen(self, dtype) -> None:
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as Cls
        self.model = Cls.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def _load_llava(self, dtype) -> None:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_internvl(self, dtype) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        kw = dict(
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        try:
            self.model = AutoModel.from_pretrained(self.model_id, use_flash_attn=True, **kw).eval()
        except Exception:
            self.model = AutoModel.from_pretrained(self.model_id, **kw).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
        self._torch = torch

    def generate(self, img_path: Path, prompt: str) -> str:
        if self.family == "internvl":
            return self._gen_internvl(img_path, prompt)
        return self._gen_hf_chat(img_path, prompt)

    def _gen_hf_chat(self, img_path: Path, prompt: str) -> str:
        import torch

        image = resize_rgb(img_path, self.max_side)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=1200, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def _gen_internvl(self, img_path: Path, prompt: str) -> str:
        pixel_values = internvl_pixels(img_path, max_num=6).to(self.model.dtype)
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device)
        gen = dict(max_new_tokens=1200, do_sample=False)
        return self.model.chat(self.tokenizer, pixel_values, prompt, gen)


def internvl_pixels(path: Path, input_size: int = 448, max_num: int = 6):
    """Official InternVL dynamic preprocess (compact)."""
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    image = Image.open(path).convert("RGB")
    w, h = image.size
    aspect = w / max(h, 1)
    target_ratios = sorted(
        {(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )

    def fit(rat):
        tw, th = input_size * rat[0], input_size * rat[1]
        r = tw / th
        return min(aspect, r) / max(aspect, r)

    best = max(target_ratios, key=fit)
    image = image.resize((input_size * best[0], input_size * best[1]))
    blocks = []
    for i in range(best[1]):
        for j in range(best[0]):
            box = (j * input_size, i * input_size, (j + 1) * input_size, (i + 1) * input_size)
            blocks.append(transform(image.crop(box)))
    thumb = transform(Image.open(path).convert("RGB").resize((input_size, input_size)))
    blocks.append(thumb)
    return torch.stack(blocks)


def summarize(ok: list[dict], cats: list[str], model: str, src: str, tgt: str, n: int) -> dict:
    def m(key, rows=ok):
        return mean([r["score"][key] for r in rows])

    summary = {
        "model": model,
        "src": src,
        "tgt": tgt,
        "n": n,
        "n_ok": len(ok),
        "translate_recall": m("translate_recall") if ok else 0,
        "translate_precision": m("translate_precision") if ok else 0,
        "over_translation": m("over_translation") if ok else 0,
        "under_translation": m("under_translation") if ok else 0,
        "pct_images_extra": m("has_extra") if ok else 0,
        "pct_images_pack_en": m("has_pack_en") if ok else 0,
        "mean_extra_spans": m("n_extra") if ok else 0,
        "by_cat": {},
    }
    for cat in cats:
        sub = [r for r in ok if r["cat"] == cat]
        if not sub:
            continue
        summary["by_cat"][cat] = {
            "n": len(sub),
            "translate_recall": mean([r["score"]["translate_recall"] for r in sub]),
            "translate_precision": mean([r["score"]["translate_precision"] for r in sub]),
            "over_translation": mean([r["score"]["over_translation"] for r in sub]),
            "pct_images_pack_en": mean([r["score"]["has_pack_en"] for r in sub]),
        }
    return summary


def run_one(model_id: str, items: list[dict], prompt: str, src: str, tgt: str, cats: list[str], max_side: int) -> Path:
    vlm = LocalVLM(model_id, max_side=max_side)
    slug = model_id.split("/")[-1].replace(" ", "_")
    dest = OUT / f"{slug}_{src}2{tgt}_n{len(items)}"
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for it in tqdm(items, desc=slug[:18]):
        img = it["folder"] / f"{src}.png"
        gt = gt_for(it["folder"], src)
        rec = {"cat": it["cat"], "stem": it["stem"], "src": src, "tgt": tgt, "gt": gt, "model": model_id}
        try:
            if not img.exists():
                raise FileNotFoundError(str(img))
            raw_txt = vlm.generate(img, prompt)
            raw = parse_json_obj(raw_txt)
            items_pred = raw.get("items") or []
            rec["raw"] = raw
            rec["score"] = {
                **score_one(items_pred, gt),
                **extra_flags(
                    [x["text"] for x in items_pred if str(x.get("decision", "")).lower().startswith("t")],
                    gt,
                ),
            }
        except Exception as e:
            rec["error"] = str(e)
        results.append(rec)
        (dest / "raw.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8"
        )
    ok = [r for r in results if "score" in r]
    summary = summarize(ok, cats, model_id, src, tgt, len(results))
    (dest / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    del vlm
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="fr")
    parser.add_argument("--tgt", default="ko")
    parser.add_argument("--n", type=int, default=24, help="0 = all items")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="qwen", help="qwen | internvl | llava | HF id")
    parser.add_argument("--all-local", action="store_true", help="run qwen, internvl, llava in sequence")
    parser.add_argument("--cat", default="cosmetic,cereal,tea")
    parser.add_argument("--max-side", type=int, default=768)
    args = parser.parse_args()

    if not ADS.exists():
        sys.exit(f"ads not found: {ADS}. Set DECISION_ADS or put ads/ next to this script.")

    cats = [c.strip() for c in args.cat.split(",") if c.strip()]
    items = list_items(cats)
    if not items:
        sys.exit(f"no items with texts.json under {ADS}")
    if args.n <= 0 or args.n >= len(items):
        sample = items
    else:
        sample = sample_items(items, args.n, args.seed)

    prompt = PROMPT.format(tgt_name=LANG_NAME.get(args.tgt, args.tgt))
    models = ["qwen", "internvl", "llava"] if args.all_local else [args.model]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"ADS={ADS} n={len(sample)}/{len(items)} models={models}", flush=True)
    for m in models:
        run_one(resolve_model(m), sample, prompt, args.src, args.tgt, cats, args.max_side)


if __name__ == "__main__":
    main()
