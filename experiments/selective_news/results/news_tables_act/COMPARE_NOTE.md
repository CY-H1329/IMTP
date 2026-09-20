# Résultats finaux — 3 modèles

# News selective translation — Decision + ACT (item all-or-nothing)

ACT success for an article = **all** gold spans correct (translate AND preserve).
Partial credit does not count as success. Span P/T rates below are diagnostic only.

## Table A. Finding (Decision probe)

| Model | n | KNOW | Decision (span) | Decision (item all) | Preserve@Dec |
|---|---:|---:|---:|---:|---:|
| InternVL3_5-8B-HF | 2843 | 75.5 | 72.3 | 0.3 | 97.6 |
| Qwen3-VL-8B-Instruct | 2841 | 76.4 | 75.1 | 8.0 | 97.0 |
| gemma-3-12b-it | 2841 | 99.4 | 79.5 | 19.5 | 99.2 |

## Table B. ACT item success (primary Δ) — all spans must be right

| Model | ACT-item unguided | ACT-item guided | Δ |
|---|---:|---:|---:|
| InternVL3_5-8B-HF | 0.0 | 0.0 | 0.0 |
| Qwen3-VL-8B-Instruct | 0.0 | 0.0 | 0.0 |
| gemma-3-12b-it | 0.0 | 0.0 | 0.0 |

## Table C. Diagnostic span breakdown (not primary)

| Model | P-ung | P-g | T-ung | T-g |
|---|---:|---:|---:|---:|
| InternVL3_5-8B-HF | 8.6 | 82.4 | 0.6 | 9.3 |
| Qwen3-VL-8B-Instruct | 10.8 | 90.6 | 1.1 | 5.3 |
| gemma-3-12b-it | 9.4 | 78.8 | 2.0 | 4.6 |

## Lecture courte
- **gemma-3-12b-it** : 1er — KNOW 99.4, Dec 79.5, DecI 19.5
- **Qwen3-VL-8B** : 2e — KNOW 76.4, Dec 75.1, DecI 8.0
- **InternVL3.5-8B** : 3e Decision — KNOW 75.5, Dec 72.3, DecI 0.3 ; guided ↑↑ Preserve (P-g ~82%)
- **ACT-item = 0%** pour les 3 (all-or-nothing + translate@gt rare)
- InternVL **2843/2843** (2 derniers IDs inclus). Qwen/Gemma **2841** (mêmes 2 IDs absents: texte avec caractères break JSONL)
