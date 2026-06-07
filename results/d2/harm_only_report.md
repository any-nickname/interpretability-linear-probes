# Qwen D2 Harm-Only Dev/Test Layer Selection

This is an interim protocol while clean `safe|refusal` rows are unavailable.
The prompt-risk label is fixed to `harm`; the target is observed Qwen behavior.

## source_matched

- Dev counts: `{'advbench|compliance': 1, 'advbench|refusal': 1, 'harmbench|compliance': 6, 'harmbench|refusal': 6, 'jailbreakbench|compliance': 8, 'jailbreakbench|refusal': 8}`
- Test counts: `{'harmbench|compliance': 5, 'harmbench|refusal': 5, 'jailbreakbench|compliance': 7, 'jailbreakbench|refusal': 7}`
- Selected layer by dev AUROC: `17`
- Selected layer dev: AUROC `0.9644`, accuracy `0.8000`, F1 `0.7692`
- Selected layer test: AUROC `0.7500`, accuracy `0.6250`, F1 `0.4706`
- Layer 18 control test: AUROC `0.7778`, accuracy `0.7083`, F1 `0.6316`
- Layer 20 control test: AUROC `0.7708`, accuracy `0.7083`, F1 `0.6957`
- Exploratory best test layer: `27` with AUROC `0.8472`, accuracy `0.6667`, F1 `0.6923`

## harm_only_max

- Dev counts: `{'advbench|compliance': 1, 'advbench|refusal': 4, 'harmbench|compliance': 9, 'harmbench|refusal': 6, 'jailbreakbench|compliance': 8, 'jailbreakbench|refusal': 8}`
- Test counts: `{'advbench|refusal': 4, 'harmbench|compliance': 9, 'harmbench|refusal': 5, 'jailbreakbench|compliance': 7, 'jailbreakbench|refusal': 7}`
- Selected layer by dev AUROC: `18`
- Selected layer dev: AUROC `0.9383`, accuracy `0.8611`, F1 `0.8387`
- Selected layer test: AUROC `0.9219`, accuracy `0.7812`, F1 `0.7586`
- Layer 18 control test: AUROC `0.9219`, accuracy `0.7812`, F1 `0.7586`
- Layer 20 control test: AUROC `0.9219`, accuracy `0.8438`, F1 `0.8485`
- Exploratory best test layer: `27` with AUROC `0.9492`, accuracy `0.8438`, F1 `0.8485`
