# Layer Selection Protocol

This document fixes the layer-selection policy for the current Qwen D1 stage.
It separates the primary reported probe from controls and exploratory views.

## Scope

- Model: `qwen`
- Probe family: `D1`, prompt-label probe (`harm` vs `safe`)
- Activation type: last-token residual activations
- Probe training data: `datasets/processed/itw_manual_d1_train.jsonl`
- Saved layer-wise probes: `data/probes/itw_manual_d1/qwen.pt`

The saved probe file contains one independent linear probe per model layer.
Layer selection chooses which already-trained layer probe to report; it does
not retrain the probe.

## Primary D1 Layer Selection

Primary protocol for the current Qwen D1 OOD-robust candidate:

1. Train one independent D1 probe per layer on `ITW train`.
2. Materialize clean OOD datasets from raw sources only.
3. Split each clean OOD dataset into stratified `50%` OOD-dev and `50%`
   OOD-test with seed `42`.
4. On OOD-dev, compute AUROC for every layer on every OOD-dev dataset.
5. Select one global layer by maximizing the unweighted mean AUROC across
   OOD-dev datasets.
6. Evaluate that fixed layer on OOD-test.

Current result:

- Selected layer: `25`
- Mean OOD-dev AUROC: `0.8659`
- Mean OOD-test AUROC: `0.8611`
- Report directory:
  `data/validation/itw_manual_d1/qwen/ood_clean/ood_dev_test_layer_selection/`

This layer should be described as:

> Qwen D1, OOD-dev-selected layer 25.

It should not be described as purely ITW-selected, because OOD-dev labels are
used for layer selection.

## Secondary Control: ITW-Selected Layer

The conservative anti-peeking control uses only the in-domain ITW eval split to
choose the layer.

Current result:

- ITW-selected layer: `12`
- Report directory:
  `data/validation/itw_manual_d1/qwen/ood_clean/fixed_layer12/`

This control answers a different question:

> If the layer is selected only from the original ITW train/eval protocol, how
> well does that fixed probe transfer to clean OOD?

It is useful as a historical baseline, but it is not the strongest current
OOD-robust D1 candidate.

## Exploratory Views

The following are exploratory and should not be used as final headline metrics:

- Choosing the best layer separately for each full OOD dataset.
- Inspecting all-layer OOD curves after seeing OOD-test results.
- Comparing many D1/D2 layer combinations without a held-out selection rule.

These views are useful for hypothesis generation, especially for understanding
where the harm/safe signal appears in the model, but they are not strict final
evaluations.

## Role of ITW Eval

In the primary OOD-dev-selected protocol, `itw_manual_d1_eval.jsonl` does not
choose the final layer. It remains:

- an in-domain diagnostic;
- a historical comparison point;
- the source of the fixed layer-12 control.

## Future D2 Protocol

When D2 is trained, use the same separation:

- train D2 probes on the D2 train split;
- choose one primary D2 layer by a declared dev rule;
- evaluate the fixed D2 layer on held-out test data;
- keep all-layer and cross-layer D1/D2 comparisons as exploratory unless they
  use a separate dev/test selection protocol.
