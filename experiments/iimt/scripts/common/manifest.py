"""Manifest loading for IIMT experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class SampleRecord:
    sample_id: str
    benchmark: str
    split: str
    scenario: str
    image_path: Path
    src_lang: str
    tgt_lang: str
    ref_text: str = ""
    ref_regions: List[Dict[str, Any]] = field(default_factory=list)
    stress_tag: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], manifest_dir: Path) -> "SampleRecord":
        img = Path(raw["image_path"])
        if not img.is_absolute():
            img = (manifest_dir / img).resolve()
        return cls(
            sample_id=str(raw["sample_id"]),
            benchmark=str(raw.get("benchmark", "custom")),
            split=str(raw.get("split", "test")),
            scenario=str(raw.get("scenario", "general")),
            image_path=img,
            src_lang=str(raw.get("src_lang", "en")),
            tgt_lang=str(raw.get("tgt_lang", "ko")),
            ref_text=str(raw.get("ref_text", "")),
            ref_regions=list(raw.get("ref_regions") or []),
            stress_tag=str(raw.get("stress_tag", "")),
            extra={k: v for k, v in raw.items() if k not in {
                "sample_id", "benchmark", "split", "scenario", "image_path",
                "src_lang", "tgt_lang", "ref_text", "ref_regions", "stress_tag",
            }},
        )


def load_manifest(path: Path) -> List[SampleRecord]:
    path = Path(path)
    records: List[SampleRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(SampleRecord.from_dict(json.loads(line), path.parent))
    return records


def iter_manifest(path: Path) -> Iterator[SampleRecord]:
    yield from load_manifest(path)
