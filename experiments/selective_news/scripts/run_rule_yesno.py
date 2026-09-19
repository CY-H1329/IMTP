#!/usr/bin/env python3
"""Stage 1b: yes/no on constructional rules + explanation + examples. No application image."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image

import os

def _sb() -> Path:
    e = os.environ.get("SELECTIVE_BENCH_ROOT")
    if e:
        return Path(e)
    here = Path(__file__).resolve()
    cand = here.parents[1]
    if (cand / "gold" / "rules.json").exists():
        return cand
    for p in (
        Path("/workspace/chanyeong/ICLR/selective_bench"),
        Path("/root/Desktop/workspace/chanyeong/ICLR/selective_bench"),
    ):
        if (p / "gold" / "rules.json").exists():
            return p
    return cand

ROOT = _sb()
ICLR = ROOT.parent
os.environ.setdefault("SELECTIVE_BENCH_ROOT", str(ROOT))
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(ICLR))
from eval_decision_local import internvl_pixels, resize_rgb  # noqa: E402

_PROMPT_CANDS = [
    ROOT / "prompts" / "rule_yesno.txt",
    ROOT / "gold" / "prompts" / "rule_yesno.txt",
]
PROMPT = next(p.read_text() for p in _PROMPT_CANDS if p.exists())
RULES = json.loads((ROOT / "gold" / "rules.json").read_text())
BLANK = ROOT / "results" / "_blank.png"
DEST = ROOT / "results" / "rule_yesno.json"

ALIASES = {
    "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL2_5-8B",
    "qwen35": "Qwen/Qwen3.5-4B",
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl35": "OpenGVLab/InternVL3_5-8B-HF",
    "qwen38": "Qwen/Qwen3.8-27B",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "qwen38-27b": "Qwen/Qwen3.8-27B",
}


def parse_yn(text: str):
    t = re.sub(r"[^A-Z]+", " ", (text or "").strip().upper())
    for tok in t.split():
        if tok in ("YES", "NO"):
            return tok
    if "YES" in t:
        return "YES"
    if "NO" in t:
        return "NO"
    return None


def all_rules():
    out = []
    for split in ("ads", "video", "news"):
        for r in RULES[split]:
            out.append({**r, "split": split})
    return out


def fill_prompt(rule: dict) -> str:
    return PROMPT.replace("{rule}", rule["rule"]).replace("{examples}", rule.get("examples", ""))


def family_of(model_id: str) -> str:
    m = model_id.lower().replace("_", "-")
    # Qwen3.8 is a native VLM (not Qwen3-VL). Match before qwen3-vl / qwen3.5.
    if "qwen3.8" in m or "qwen38" in m:
        return "qwen38"
    if "qwen3-vl" in m or "qwen3vl" in m:
        return "qwen3vl"
    if "qwen3.5" in m or "qwen35" in m:
        return "qwen35"
    if "internvl3-5" in m or "internvl3.5" in m or ("internvl" in m and m.endswith("-hf")):
        return "hf_vlm"
    if "internvl" in m:
        return "internvl"
    if "gemma-3" in m or "gemma3" in m:
        return "hf_vlm"
    if "qwen2-vl" in m and "2.5" not in m:
        return "qwen2"
    return "qwen25"


class LocalVLM:
    def __init__(self, model_id: str, max_side: int, max_new: int):
        self.model_id = model_id
        self.family = family_of(model_id)
        self.max_side = max_side
        self.max_new = max_new
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load()

    def _device(self):
        return next(self.model.parameters()).device

    def _load(self) -> None:
        import torch

        dtype = torch.bfloat16
        print(f"loading {self.model_id} family={self.family}", flush=True)
        if self.family == "internvl":
            from transformers import AutoModel, AutoTokenizer

            kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto")
            try:
                self.model = AutoModel.from_pretrained(self.model_id, use_flash_attn=True, **kw).eval()
            except Exception:
                self.model = AutoModel.from_pretrained(self.model_id, **kw).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
            return
        from transformers import AutoProcessor

        if self.family in ("qwen35", "qwen38"):
            # Qwen3.8-27B is a native VLM on the Qwen3.5 architecture
            # (config.architectures = Qwen3_5ForConditionalGeneration).
            from transformers import Qwen3_5ForConditionalGeneration as Cls
        elif self.family == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration as Cls
        elif self.family == "hf_vlm":
            from transformers import AutoModelForImageTextToText as Cls
        elif self.family == "qwen2":
            from transformers import Qwen2VLForConditionalGeneration as Cls
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        kw = dict(torch_dtype=dtype, device_map="auto", trust_remote_code=True)
        if self.family not in ("qwen35", "qwen38", "hf_vlm"):
            kw["attn_implementation"] = "sdpa"
        self.model = Cls.from_pretrained(self.model_id, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def _to_device(self, inputs):
        device = self._device()
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _decode_new(self, out, inputs) -> str:
        in_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs["input_ids"]
        trimmed = out[:, in_ids.shape[1] :]
        text_out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        if "</think>" in text_out:
            text_out = text_out.split("</think>", 1)[-1].strip()
        return text_out

    def generate(self, img_path: Path, prompt: str, max_new: int | None = None) -> str:
        ntok = self.max_new if max_new is None else max_new
        if self.family == "internvl":
            pixel_values = internvl_pixels(img_path, max_num=4).to(self.model.dtype)
            pixel_values = pixel_values.to(self._device())
            cfg = dict(max_new_tokens=ntok, do_sample=False)
            try:
                return self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    cfg,
                    history=None,
                    return_history=False,
                )
            except TypeError:
                return self.model.chat(self.tokenizer, pixel_values, prompt, cfg)
        import torch

        image = resize_rgb(img_path, self.max_side)
        extra = {"enable_thinking": False} if self.family in ("qwen35", "qwen38") else {}
        if self.family in ("qwen3vl", "hf_vlm", "qwen38"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    **extra,
                )
            except TypeError:
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True, **extra)
                inputs = self.processor(text=text, images=[image], return_tensors="pt")
        else:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, **extra)
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
        inputs = self._to_device(inputs)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=ntok, do_sample=False)
        return self._decode_new(out, inputs)

    def close(self) -> None:
        import torch

        del self.model
        self.model = None
        torch.cuda.empty_cache()


def merge_write(results: list) -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    prev = []
    if DEST.exists():
        try:
            prev = json.loads(DEST.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                prev = prev.get("models", [])
        except Exception:
            prev = []
    run_ids = {m["model"] for m in results}
    prev = [m for m in prev if m.get("model") not in run_ids]
    out = {
        "prompt_version": "explained+examples",
        "models": prev + results,
    }
    DEST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", DEST)
    for m in out["models"]:
        print(f"SUMMARY {m['model'].split('/')[-1]} {m['n_ok']}/{m['n']}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen25", "internvl", "qwen35"])
    ap.add_argument("--out", type=Path, default=None, help="override DEST json path")
    args = ap.parse_args()
    global DEST
    if args.out:
        DEST = args.out
    BLANK.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (448, 448), "white").save(BLANK)
    rows = all_rules()
    results = []
    from progress_util import print_progress

    for spec in args.models:
        mid = ALIASES.get(spec, spec)
        vlm = LocalVLM(mid, max_side=448, max_new=16)
        recs = []
        total = len(rows)
        for i, r in enumerate(rows, 1):
            raw = vlm.generate(BLANK, fill_prompt(r))
            yn = parse_yn(raw)
            recs.append(
                {
                    "id": r["id"],
                    "split": r["split"],
                    "expect": r["expect"],
                    "pred": yn,
                    "ok": yn == r["expect"],
                    "raw": (raw or "")[:300],
                }
            )
            print_progress(
                i,
                total,
                prefix=mid.split("/")[-1],
                extra=f"{r['id']} expect={r['expect']} pred={yn}",
            )
        vlm.close()
        results.append(
            {
                "model": mid,
                "prompt_version": "explained+examples",
                "n_ok": sum(x["ok"] for x in recs),
                "n": len(recs),
                "items": recs,
            }
        )
    merge_write(results)


if __name__ == "__main__":
    main()
