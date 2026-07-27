"""Standard output contract for all baselines."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def save_outputs(
    out_dir: Path,
    *,
    pred_text: str,
    pred_image_path: Optional[Path] = None,
    reocr: Optional[Dict[str, Any]] = None,
    baseline_id: str,
    baseline_name: str,
    record_meta: Dict[str, Any],
    latency_sec: float,
    model_version: str = "",
    status: str = "ok",
    error: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "pred_text.txt").write_text(pred_text or "", encoding="utf-8")

    if pred_image_path and pred_image_path.exists():
        target = out_dir / "pred_image.png"
        if pred_image_path.resolve() != target.resolve():
            target.write_bytes(pred_image_path.read_bytes())

    if reocr is not None:
        write_json(out_dir / "reocr.json", reocr)

    meta = {
        "baseline_id": baseline_id,
        "baseline_name": baseline_name,
        "status": status,
        "error": error,
        "latency_sec": round(latency_sec, 4),
        "model_version": model_version,
        "git_hash": git_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **record_meta,
    }
    if extra:
        meta["extra"] = extra
    write_json(out_dir / "meta.json", meta)
    return out_dir


def load_meta(out_dir: Path) -> Dict[str, Any]:
    return json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))


def collect_pred_roots(outputs_root: Path, baseline_id: str, benchmark: str) -> List[Path]:
    base = outputs_root / baseline_id / benchmark
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / "meta.json").exists())
