# Qwen D2 OOD-Dev/Test Layer Selection

- Probe: ITW-trained Qwen D2 response-behavior probe
- Positive class: `refusal`
- Negative class: `compliance`
- OOD-dev: `50` refusal / `50` compliance
- OOD-test: `50` refusal / `50` compliance
- Selected layer by OOD-dev AUROC: `18`
- Selected layer OOD-dev AUROC/accuracy/F1: `0.9484` / `0.7900` / `0.7470`
- Selected layer OOD-test AUROC/accuracy/F1: `0.9116` / `0.8200` / `0.8085`
- ITW-eval-selected layer `20` OOD-test AUROC: `0.8580`
- Exploratory best OOD-test layer: `27` with AUROC `0.9284`

The exploratory best OOD-test layer is not used for selection.
