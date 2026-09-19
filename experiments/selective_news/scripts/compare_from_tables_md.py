#!/usr/bin/env python3
from pathlib import Path
import sys
p = Path(sys.argv[1] if len(sys.argv)>1 else "results/news_tables_act/tables.md")
text = p.read_text()
out = Path(sys.argv[2] if len(sys.argv)>2 else "results/news_tables_act/compare_models.md")
out.write_text("# InternVL vs Qwen3 (and others)\n\n" + text + "\n", encoding="utf-8")
print(out.read_text())
