# Qwen ITW Manual D2-Core Summary

## Scope

- Model: `Qwen/Qwen3-1.7B`
- D2 target: response behavior
- Positive class: `refusal`
- Negative class: `compliance`
- Excluded from first binary run: `partial`, `unclear`
- Activations: existing Qwen last-token prompt activations from the ITW D1 split
- Train prompts: `datasets/processed/itw_manual_d1_train.jsonl`
- Eval prompts: `datasets/processed/itw_manual_d1_eval.jsonl`

## Response Collection

Responses were generated directly with local TransformerLens / HookedTransformer,
not with LM Studio.

- Decoding: greedy, `do_sample=False`, `temperature=0.0`
- Max new tokens: `128`
- Train responses: `236`
- Eval responses: `60`

Response files:

- `data/responses/itw_manual_d2/qwen/itw_manual_d1_train_responses.jsonl`
- `data/responses/itw_manual_d2/qwen/itw_manual_d1_eval_responses.jsonl`

## Manual Labels

Final behavior labels are manually assigned. Scripts only materialize the manual
decision table and validate counts.

Label counts:

| Split | Compliance | Refusal | Partial |
|---|---:|---:|---:|
| train | 214 | 13 | 9 |
| eval | 53 | 5 | 2 |
| all | 267 | 18 | 11 |

Usable binary records after excluding `partial`:

- train: `227` (`13` refusal / `214` compliance)
- eval: `58` (`5` refusal / `53` compliance)

## D2 Probe

One independent linear probe was trained per layer. LogisticRegression used
`class_weight="balanced"` because refusal labels are rare.

Best eval layer:

- Layer: `20`
- Eval AUROC: `0.8906`
- Eval accuracy: `0.9310`
- Eval F1: `0.5000`

Top eval AUROC layers:

| Layer | Eval AUROC | Eval accuracy | Eval F1 |
|---:|---:|---:|---:|
| 20 | 0.8906 | 0.9310 | 0.5000 |
| 12 | 0.8868 | 0.9483 | 0.6667 |
| 13 | 0.8642 | 0.9483 | 0.6667 |
| 10 | 0.8491 | 0.9138 | 0.4444 |
| 21 | 0.8491 | 0.9483 | 0.6667 |

Outputs:

- Layer metrics: `data/probes/itw_manual_d2/qwen_layers.csv`
- Probe payload: `data/probes/itw_manual_d2/qwen.pt`
- Probe summary: `data/probes/itw_manual_d2/qwen_summary.json`
- AUROC plot: `data/probes/itw_manual_d2/figures/qwen_d2_auroc_by_layer.svg`

## Interpretation

- D2-core has a learnable signal on ITW eval: best AUROC is `0.8906`.
- The result is promising but fragile: eval contains only `5` refusal examples.
- Train AUROC is `1.0` across layers, so overfitting risk is real.
- This is enough to justify a small manually labeled OOD-test, but not enough
  for strong claims about D2 generalization.
