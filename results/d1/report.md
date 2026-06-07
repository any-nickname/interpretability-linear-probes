# OOD-Dev Layer Selection and OOD-Test Evaluation

The probes are still trained only on the ITW train split. OOD-dev is used
only to choose one global layer. OOD-test is then used for the held-out
evaluation of that fixed layer.

- Selection rule: `single global layer maximizing unweighted mean AUROC across OOD-dev datasets`
- Dev fraction per OOD dataset: `0.5`
- Seed: `42`
- Selected layer: `25`
- Mean OOD-dev AUROC at selected layer: `0.8659`
- Mean OOD-test AUROC at selected layer: `0.8611`

## Selected Layer Test Metrics

| Dataset | Test n | Test harm | AUROC | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| `ood_advbench_alpaca` | 520 | 260 | 0.9553 | 0.8885 | 0.8934 |
| `ood_harmbench_alpaca` | 400 | 200 | 0.7942 | 0.6800 | 0.6000 |
| `ood_jailbreakbench` | 100 | 50 | 0.7968 | 0.6700 | 0.7227 |
| `ood_xstest` | 225 | 100 | 0.8982 | 0.8133 | 0.7586 |

## OOD-Test Text Baselines

| Dataset | TF-IDF AUROC | Length/token AUROC | Qwen-token AUROC |
|---|---:|---:|---:|
| `ood_advbench_alpaca` | 0.6655 | 0.5978 | 0.3897 |
| `ood_harmbench_alpaca` | 0.4779 | 0.3559 | 0.6326 |
| `ood_jailbreakbench` | 0.6172 | 0.3404 | 0.6450 |
| `ood_xstest` | 0.5805 | 0.5713 | 0.4341 |

## Interpretation

- This is stricter than selecting the best layer on each full OOD dataset.
- It is still OOD-informed model selection, so final claims should call it
  `OOD-dev selected`, not `ITW-selected`.
- ITW eval is not used in this layer choice; it remains an in-domain
  diagnostic and historical comparison point.
