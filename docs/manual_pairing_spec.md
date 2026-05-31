# In-the-Wild manual pairs spec

Goal: test a stricter dataset design where the only harmful source is
In-the-Wild, and every safe example is manually written as an independent safe
pair for a specific In-the-Wild prompt.

This branch deliberately does not use Alpaca, XSTest, AdvBench, D1-hard-semantic,
or `safe_close` as safe prompt sources.

## Unit

```text
in_the_wild_harm_ref + manually_written_safe_pair
```

The safe pair is not a generic benign prompt. It should be manually written and
should preserve enough of the source prompt's outer form to make simple
length/style/lexical shortcuts less reliable: role cards, fictional scenarios,
rule blocks, developer/eval framing, style switches, transcripts, or long-form
instructions.

## Manual Pair Criteria

- Preserve broad outer complexity when useful: multiline, role setup, formatting
  constraints, transcript feel, or fictional frame.
- Do not make every safe prompt an "analysis of jailbreaks" prompt.
- Avoid a repeated global safe template.
- It is acceptable for safe prompts to contain words such as model, response,
  user, rules, system, ChatGPT, always, never, no, or prompt when they are used
  in a benign request.
- The safe prompt must be naturally answerable by an aligned assistant.
- The safe prompt must not request bypassing policies, unrestricted compliance,
  or operational wrongdoing.

## Active Track

`user_manual`: the user writes or approves safe pairs in small batches. Codex
may propose candidates, audit, validate, and materialize the dataset, but the
active safe prompts should be explicitly approved before use.

## Files

- Source In-the-Wild prompts:
  `datasets/processed/in_the_wild_source_prompts.jsonl`
- Finalized 9-pair pilot candidate:
  `datasets/pairs/in_the_wild_manual_safe_pairs_final_0000_0009.jsonl`
- Earlier user safe-pair draft:
  `datasets/pairs/in_the_wild_manual_safe_pairs_user_0000_0009.jsonl`

`itw_0007` is excluded from the pair file because it is tracked as a
`safe_hard_negative` source row rather than a harm prompt with a safe pair.

## Current Minimal Schemas

Source rows:

```json
{"id": "itw_0000", "prompt": "...", "label": "harm", "source": "in_the_wild_jb", "meta": {"platform": "...", "origin": "...", "date": "..."}}
```

Safe-pair rows:

```json
{"harm_id": "itw_0000", "safe_prompt": "..."}
```

Raw harmful prompts are not duplicated in the safe-pair ref files; they are
resolved from `datasets/processed/in_the_wild_source_prompts.jsonl` by
`harm_id`.

Legacy processed splits, D2 pair files, and the Codex diagnostic pilot are kept
under `trash/datasets/` and should not define the schema for this branch.
