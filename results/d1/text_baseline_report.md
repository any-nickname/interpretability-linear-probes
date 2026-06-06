# Exact Train/Eval Text Baselines

- Train records: `236` (`118` pairs)
- Eval records: `60` (`30` pairs)
- Token count source: `Qwen/Qwen3-1.7B chat template`

## Metrics

- TF-IDF LogisticRegression AUROC: `0.5550`
- TF-IDF LogisticRegression accuracy/F1: `0.5333` / `0.6000`
- Length/token LogisticRegression AUROC: `0.5333`
- Length/token LogisticRegression accuracy/F1: `0.5000` / `0.5455`
- Raw char length AUROC (longer = harm): `0.4717`
- Raw word length AUROC (longer = harm): `0.4694`
- Raw Qwen token count AUROC (more tokens = harm): `0.4611`
- Majority accuracy: `0.5000`
