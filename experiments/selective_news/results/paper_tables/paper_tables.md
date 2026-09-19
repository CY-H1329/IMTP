# News selective translation — Decision + ACT (item all-or-nothing)

ACT success for an article = **all** gold spans correct (translate AND preserve).
Partial credit does not count as success. Span P/T rates below are diagnostic only.

## Table A. Finding (Decision probe)

| Model | n | KNOW | Decision (span) | Decision (item all) | Preserve@Dec |
|---|---:|---:|---:|---:|---:|
| InternVL3_5-8B-HF | 2790 | 75.5 | 72.3 | 0.3 | 97.7 |
| Qwen3-VL-8B-Instruct | 2841 | 76.4 | 75.1 | 8.0 | 97.0 |

## Table B. ACT item success (primary Δ) — all spans must be right

| Model | ACT-item unguided | ACT-item guided | Δ |
|---|---:|---:|---:|
| InternVL3_5-8B-HF | 0.0 | 0.0 | 0.0 |
| Qwen3-VL-8B-Instruct | 0.0 | 0.0 | 0.0 |

## Table C. Diagnostic span breakdown (not primary)

| Model | P-ung | P-g | T-ung | T-g |
|---|---:|---:|---:|---:|
| InternVL3_5-8B-HF | 8.7 | 82.6 | 0.6 | 9.4 |
| Qwen3-VL-8B-Instruct | 10.8 | 90.6 | 1.1 | 5.3 |
