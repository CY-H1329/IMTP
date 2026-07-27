#!/usr/bin/env python3
"""Diagnose HuggingFace auth / network issues on shared JupyterLab."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    section("1) Environment token variables")
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_HUB_TOKEN",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_ENDPOINT",
    ):
        val = os.environ.get(key)
        if key.endswith("TOKEN") and val:
            print(f"  {key}=<set len={len(val)} prefix={val[:6]}...>")
        else:
            print(f"  {key}={val!r}")

    section("2) Token files on disk")
    candidates = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
        Path(os.environ.get("HF_HOME", "")) / "token" if os.environ.get("HF_HOME") else None,
        Path.cwd() / ".cache" / "huggingface" / "token",
    ]
    for p in candidates:
        if p is None:
            continue
        exists = p.is_file()
        print(f"  {p}: exists={exists}")
        if exists:
            try:
                raw = p.read_text(encoding="utf-8").strip()
                print(f"    len={len(raw)} prefix={raw[:8]!r}")
            except Exception as e:
                print(f"    read_error={e}")

    section("3) huggingface_hub.get_token()")
    try:
        from huggingface_hub import get_token

        t = get_token()
        if t:
            print(f"  get_token()=<set len={len(t)} prefix={t[:6]}...>")
        else:
            print("  get_token()=None  (good for anonymous public download)")
    except Exception as e:
        print(f"  ERROR: {e}")

    section("4) ~/.netrc huggingface entries")
    netrc = Path.home() / ".netrc"
    if netrc.is_file():
        text = netrc.read_text(encoding="utf-8", errors="replace")
        if "huggingface" in text.lower():
            print("  WARNING: ~/.netrc contains huggingface — this often causes 401")
            for i, line in enumerate(text.splitlines(), 1):
                if "huggingface" in line.lower() or "login" in line.lower() or "password" in line.lower():
                    # redact password values
                    low = line.lower().strip()
                    if low.startswith("password"):
                        print(f"  L{i}: password ****")
                    else:
                        print(f"  L{i}: {line}")
        else:
            print("  no huggingface mention in ~/.netrc")
    else:
        print("  no ~/.netrc")

    section("5) Anonymous HTTP to HuggingFace (no Authorization header)")
    url = "https://huggingface.co/Helsinki-NLP/opus-mt-en-ko/resolve/main/config.json"
    try:
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        # Explicitly do NOT add Authorization
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"  HEAD {url}")
            print(f"  status={resp.status} ok (anonymous works)")
    except Exception as e:
        print(f"  HEAD failed: {type(e).__name__}: {e}")
        print("  → Cluster may block anonymous HF, or network/proxy issue.")

    section("6) huggingface_hub with token=False")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="Helsinki-NLP/opus-mt-en-ko",
            filename="config.json",
            token=False,
        )
        print(f"  OK downloaded: {path}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    section("7) Recommendation")
    print(
        """
If (5) anonymous HEAD works but (6) fails:
  → stale token still attached. Delete token files + unset env vars.

If (5) AND (6) both fail with 401:
  → this JupyterLab network requires a VALID HF token for all HF access.
  → run:  huggingface-cli login
  → or download model elsewhere and set IIMT_MARIAN_DIR

If (4) shows ~/.netrc huggingface:
  → remove those lines (backup first):  cp ~/.netrc ~/.netrc.bak
"""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
