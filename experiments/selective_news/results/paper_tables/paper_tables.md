# News selective translation — Decision + ACT (item all-or-nothing)

ACT success for an article = **all** gold spans correct (translate AND preserve).
Partial credit does not count as success. Span P/T rates below are diagnostic only.

## Table A. Finding (Decision probe)

| Model | n | KNOW | Decision (span) | Decision (item all) | Preserve@Dec |
|---|---:|---:|---:|---:|---:|
| InternVL3_5-8B-HF | 2843 | 75.5 | 72.3 | 0.3 | 97.6 |
| Qwen3-VL-8B-Instruct | 2841 | 76.4 | 75.1 | 8.0 | 97.0 |
| Qwen3.8-27B | 2843 | 75.1 | 78.9 | 15.8 | 99.7 |
| gemma-3-12b-it | 2841 | 99.4 | 79.5 | 19.5 | 99.2 |
| gemma-3-27b-it | 2843 | 95.4 | 78.6 | 16.5 | 99.7 |
| pixtral-12b | 2843 | 79.8 | 81.7 | 28.5 | 98.5 |

## Table B. ACT item success (primary Δ) — all spans must be right

| Model | ACT-item unguided | ACT-item guided | Δ |
|---|---:|---:|---:|
| InternVL3_5-8B-HF | 0.0 | 0.0 | 0.0 |
| Qwen3-VL-8B-Instruct | 0.0 | 0.0 | 0.0 |
| Qwen3.8-27B | 0.0 | 0.1 | 0.1 |
| gemma-3-12b-it | 0.0 | 0.0 | 0.0 |
| gemma-3-27b-it | 0.0 | 0.0 | 0.0 |
| pixtral-12b | 0.0 | 0.0 | 0.0 |

## Table C. Diagnostic span breakdown (not primary)

| Model | P-ung | P-g | T-ung | T-g |
|---|---:|---:|---:|---:|
| InternVL3_5-8B-HF | 8.6 | 82.4 | 0.6 | 9.3 |
| Qwen3-VL-8B-Instruct | 10.8 | 90.6 | 1.1 | 5.3 |
| Qwen3.8-27B | 6.9 | 85.1 | 2.6 | 22.2 |
| gemma-3-12b-it | 9.4 | 78.8 | 2.0 | 4.6 |
| gemma-3-27b-it | 7.9 | 85.4 | 1.7 | 23.2 |
| pixtral-12b | 0.7 | 91.9 | 0.7 | 17.2 |
