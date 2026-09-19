# News selective translation — Decision + ACT (item all-or-nothing)

ACT success for an article = **all** gold spans correct (translate AND preserve).
Partial credit does not count as success. Span P/T rates below are diagnostic only.

## Table A. Finding (Decision probe)

| Model | n | KNOW | Decision (span) | Decision (item all) | Preserve@Dec |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 4 | 75.0 | 75.0 | 0.0 | 100.0 |

## Table B. ACT item success (primary Δ) — all spans must be right

| Model | ACT-item unguided | ACT-item guided | Δ |
|---|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 0.0 | 0.0 | 0.0 |

## Table C. Diagnostic span breakdown (not primary)

| Model | P-ung | P-g | T-ung | T-g |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 75.0 | 100.0 | 0.0 | 8.3 |
