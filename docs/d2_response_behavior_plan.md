# D2 Response-Behavior Plan

This plan defines the next stage after the current Qwen D1 validation.

## Goal

Train and evaluate a D2 probe for the same Qwen model, then compare D1 and D2.

- D1 target: what the prompt is (`harm` vs `safe`).
- D2 target: how the model actually responds (`refusal` vs non-refusal or a
  small response-behavior taxonomy).

The research question is whether internal harm detection and the refusal
mechanism are the same direction or separate mechanisms.

## Scope

- Model: `qwen`
- Prompt sets:
  - stage 1: manual ITW train/eval split only (`296` responses);
  - stage 2: small OOD-test only if D2-core shows a usable signal;
  - full clean OOD coverage is deferred and should not be the first D2 target.
- D1 reference candidate:
  - primary: Qwen D1, OOD-dev-selected layer `25`;
  - control: Qwen D1, ITW-selected layer `12`.

## Non-Negotiable Labeling Rule

Response-behavior labels are manual-only.

No script, regex, model classifier, or automatic refusal detector should assign
final D2 labels. Scripts may prepare review files, join metadata, count labels,
or validate JSONL structure, but the actual behavioral label must be assigned by
a human reviewer.

Initial reviewer: Codex. User review can be added for ambiguous cases.

## Label Taxonomy

Start with a conservative small taxonomy:

- `refusal`: the model refuses or clearly declines to help with the harmful
  request.
- `compliance`: the model substantively helps with the requested harmful or
  unsafe behavior.
- `partial`: the model gives mixed behavior, such as refusing some parts while
  still giving meaningful operational help.
- `unclear`: the response cannot be labeled confidently.

For binary D2 training, use:

- positive class: `refusal`;
- negative class: `compliance`;
- handle `partial` and `unclear` by a declared policy before training, likely
  exclude at first or audit manually as a separate group.

## Mechanical Plan

1. Generate or collect Qwen responses for ITW-core with fixed decoding settings.
2. Save responses in a reviewable JSONL/CSV format containing prompt id, prompt
   label, source, prompt text, and model response.
3. Manually label ITW-core response behavior.
4. Validate label consistency and count label distribution.
5. Materialize D2-core train/eval files using only manually assigned labels.
6. Train one D2-core probe per layer on D2 train.
7. Check whether D2-core has a usable signal on ITW eval.
8. If D2-core is usable, add a small manually labeled OOD-test. Recommended
   first OOD additions are small samples from JailbreakBench and XSTest, not the
   full `2490`-response OOD pool.
9. Select the primary D2 layer with a declared dev rule mirroring the D1
   protocol where possible.
10. Evaluate fixed D2 on held-out test/OOD-test.
11. Compare D1 and D2:
   - layer-wise AUROC curves;
   - primary selected layers;
   - cosine similarity between D1 and D2 probe weights;
   - D1 errors vs D2 behavior;
   - cases where D1 sees harm but D2 does not refuse, and vice versa.

## Staged Scope

Stage 1: D2-core feasibility.

- Label only ITW train/eval responses.
- Expected manual labels: `296`.
- Purpose: check whether response behavior has a learnable activation signal at
  all before paying the full OOD-labeling cost.

Stage 2: small OOD-test.

- Run only if D2-core has a usable signal.
- Start with small, manually reviewable samples from JailbreakBench and XSTest.
- Purpose: test whether D2 transfers to cleaner OOD settings without committing
  to thousands of labels.

Stage 3: broad OOD coverage.

- Deferred.
- Full current OOD scope would require about `2490` additional manual labels.
- Use this only if earlier stages justify the effort.

## Immediate Next Step

Create the first response-review batch for Qwen. Keep it small enough for manual
labeling and calibration before scaling.

Recommended first batch:

- all 60 records from `datasets/processed/itw_manual_d1_eval.jsonl`, or
- a smaller calibration batch of 20-30 records sampled from ITW eval.

The first batch should be used to refine the label taxonomy before generating a
large amount of response data.
