# Label Permutation Sanity Check

- Model: `qwen`
- Permutations: `30`
- Overall mean AUROC: `0.4932`
- Overall std AUROC: `0.0587`
- Overall min/max AUROC: `0.3067` / `0.7011`
- Best mean layer: `18` (`0.5019`)

Expected result: AUROC near chance. A high value here would suggest label leakage or another pipeline issue.
