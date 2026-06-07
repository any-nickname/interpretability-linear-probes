# Clean OOD Validation Summary

This OOD block intentionally uses clean raw-derived datasets only. Legacy
processed OOD artifacts under `trash/` were not reused.

## Inputs

- Train split for text baselines: `datasets/processed/itw_manual_d1_train.jsonl`
- Trained activation probe: `data/probes/itw_manual_d1/qwen.pt`
- OOD JSONL directory: `datasets/processed/ood_clean/`
- OOD activations: `data/activations/ood_clean/qwen/`

## Clean OOD Sets

| Dataset | Records | Harm | Safe | Notes |
|---|---:|---:|---:|---|
| `ood_jailbreakbench` | 200 | 100 | 100 | JailbreakBench harmful/benign goals |
| `ood_xstest` | 450 | 200 | 250 | `contrast_*` types treated as harm, other XSTest prompt types as safe |
| `ood_advbench_alpaca` | 1040 | 520 | 520 | AdvBench harm plus seeded Alpaca safe sample |
| `ood_harmbench_alpaca` | 800 | 400 | 400 | HarmBench harm plus seeded Alpaca safe sample |

## Activation Probe OOD Metrics

| Dataset | Best layer | Best AUROC | Accuracy | F1 | Layer 0 AUROC | Layer 12 AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `ood_advbench_alpaca` | 27 | 0.9723 | 0.8192 | 0.8459 | 0.8796 | 0.7990 |
| `ood_harmbench_alpaca` | 26 | 0.8093 | 0.7125 | 0.6849 | 0.6701 | 0.7274 |
| `ood_jailbreakbench` | 23 | 0.8156 | 0.7350 | 0.7488 | 0.6030 | 0.5432 |
| `ood_xstest` | 25 | 0.8854 | 0.8111 | 0.7550 | 0.5227 | 0.6183 |

## Text/Length OOD Baselines

| Dataset | TF-IDF AUROC | Length/token AUROC | Char length AUROC | Word length AUROC | Qwen-token AUROC |
|---|---:|---:|---:|---:|---:|
| `ood_advbench_alpaca` | 0.6571 | 0.5967 | 0.4987 | 0.5006 | 0.3942 |
| `ood_harmbench_alpaca` | 0.4729 | 0.3460 | 0.6709 | 0.6575 | 0.6452 |
| `ood_jailbreakbench` | 0.6195 | 0.3725 | 0.6544 | 0.6150 | 0.6080 |
| `ood_xstest` | 0.6057 | 0.5445 | 0.4662 | 0.4937 | 0.4664 |

## Interpretation

- The trained D1 activation probe generalizes above text baselines on all four
  clean OOD checks.
- The strongest OOD layers are late (`23`-`27`), while the in-split best layer
  was `12`. This means the current layer-wise story should not be summarized as
  "layer 12 is the semantic harm layer" without more checks.
- `ood_jailbreakbench` and `ood_xstest` are the most informative primary OOD
  checks here because they avoid the strongest broad-source mismatch of
  AdvBench/HarmBench harm versus Alpaca safe.
- The broad AdvBench/HarmBench plus Alpaca tests remain useful stress tests, but
  they are more style/source-confounded and should be interpreted cautiously.

Detailed outputs:

- Layer-selection protocol:
  `data/validation/itw_manual_d1/qwen/layer_selection_protocol.md`
- Probe metrics: `data/validation/itw_manual_d1/qwen/ood_clean/probe_eval/`
- Text baselines: `data/validation/itw_manual_d1/qwen/ood_clean/text_baselines/`
- Fixed layer-12 view: `data/validation/itw_manual_d1/qwen/ood_clean/fixed_layer12/`
- OOD-dev/test layer selection: `data/validation/itw_manual_d1/qwen/ood_clean/ood_dev_test_layer_selection/`

## OOD-Dev/Test Layer Selection

The stricter OOD-informed layer-selection protocol splits each clean OOD set
stratified into `50%` dev and `50%` test. The saved ITW-trained layer probes are
not retrained. OOD-dev chooses one global layer by unweighted mean AUROC across
datasets; OOD-test then evaluates that fixed layer.

- Selected layer: `25`
- Mean OOD-dev AUROC: `0.8659`
- Mean OOD-test AUROC: `0.8611`

| Dataset | OOD-test AUROC | Accuracy | F1 | TF-IDF test AUROC |
|---|---:|---:|---:|---:|
| `ood_advbench_alpaca` | 0.9553 | 0.8885 | 0.8934 | 0.6655 |
| `ood_harmbench_alpaca` | 0.7942 | 0.6800 | 0.6000 | 0.4779 |
| `ood_jailbreakbench` | 0.7968 | 0.6700 | 0.7227 | 0.6172 |
| `ood_xstest` | 0.8982 | 0.8133 | 0.7586 | 0.5805 |
