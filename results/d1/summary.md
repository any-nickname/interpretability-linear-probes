# Qwen D1 Validation Summary

Scope: current 148-pair manual In-the-Wild D1 split.

## Exact Text Baselines

Location: `data/validation/itw_manual_d1/qwen/text_baselines/`

- TF-IDF LogisticRegression AUROC: `0.5550`
- Length/token LogisticRegression AUROC: `0.5333`
- Raw char/word/Qwen-token-count AUROC (longer/more = harm): `0.4717` / `0.4694` / `0.4611`
- Majority accuracy: `0.5000`

Interpretation: text, length, and token-count baselines on the exact D1 split remain far below the last-token activation probe.

## Label Permutation

Location: `data/validation/itw_manual_d1/qwen/label_permutation/`

- Permutations: `30`
- Overall AUROC mean/std: `0.4932` / `0.0587`
- Best mean layer: `18` with AUROC `0.5019`

Interpretation: shuffled train labels collapse to chance, so the high real-label result is not explained by an obvious label leak.

## Last-Token Activation Probe

Location: `data/probes/itw_manual_d1/`

- Best layer: `12`
- Best AUROC: `0.9533`
- Layer 0 AUROC: `0.9244`

Interpretation: strong result, but high layer-0 separability requires caution.

## Mean-Pooled Activation Probe

Location: `data/probes/itw_manual_d1_mean_pool/`

- Best layer: `9`
- Best AUROC: `0.8167`
- Layer 0 AUROC: `0.7211`

Interpretation: mean-pooling is substantially weaker than last-token pooling, suggesting the strongest D1 signal is concentrated in the final prompt state rather than uniformly across token positions.

## Layer-0 Diagnostics

Location: `data/validation/itw_manual_d1/qwen/layer0_diagnostics/`

- Layer 0 AUROC/accuracy/F1: `0.9244` / `0.8667` / `0.8750`
- Layer 12 AUROC/accuracy/F1: `0.9533` / `0.8833` / `0.8889`
- Layer 0 errors: `8`
- Correlation between layer-0 harm score and TF-IDF harm score: Pearson `0.0028`, Spearman `0.0606`
- Correlation between layer-0 harm score and length/token baselines is weak: char length Spearman `-0.0852`, Qwen token-count Spearman `-0.1048`

Interpretation: layer-0 scores are not simply monotonic with TF-IDF or length/token count. Manual inspection is still needed because several confident layer-0 errors involve safe prompts that intentionally preserve jailbreak-like surface language.
