# Qwen D1 vs D2 Comparison

## Current selected probes

- D1 prompt-label probe: OOD-dev-selected layer `25`
- D2 response-behavior probe: OOD-dev-selected layer `18`
- D2 ITW-eval-selected control layer: `20`

## Prompt label vs response behavior

| Dataset | n | accuracy if refusal=harm | MCC | AUROC prompt-label->behavior |
|---|---:|---:|---:|---:|
| ITW-core | 285 | 0.5123 | 0.0064 | 0.5066 |
| OOD-balanced-200 | 200 | 0.9800 | 0.9608 | 0.9800 |

## Score correlation on shared OOD-balanced prompts

- Selected D1 layer `25` harm-score vs selected D2 layer `18` refusal-score on OOD-test: Pearson `0.6082`, Spearman `0.7127`.

## Same-layer cosine

- Same-layer cosine at D2-selected layer `18`: `0.1153`.
- Same-layer cosine at D1-selected layer `25`: `0.0853`.
- Cross-layer cosine between D1 layer 25 and D2 layer 18 is intentionally not reported.

## Error overlap on shared OOD-balanced prompts

| Split | n | D1 errors | D2 errors | Both errors | Either errors |
|---|---:|---:|---:|---:|---:|
| ood_dev | 100 | 21 | 21 | 5 | 37 |
| ood_test | 100 | 16 | 18 | 4 | 30 |

Full CSV outputs are saved next to this report.
