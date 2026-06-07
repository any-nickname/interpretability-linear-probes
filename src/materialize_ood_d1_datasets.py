"""
Materialize clean OOD datasets for evaluating the manual ITW D1 probe.

Only raw sources are used here. Legacy processed OOD artifacts under trash/ are
intentionally ignored.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path


ROOT = Path("C:/Users/Vladi/Desktop/projs/interpetability")
DEPS = ROOT / ".deps" / "python"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def alpaca_prompt(row: dict) -> str:
    instruction = row["instruction"].strip()
    input_text = row.get("input", "").strip()
    if input_text:
        return f"{instruction}\n\nInput: {input_text}"
    return instruction


def build_jailbreakbench() -> list[dict]:
    from datasets import load_from_disk

    ds = load_from_disk(str(ROOT / "datasets" / "raw" / "jailbreakbench"))
    rows = []
    for label, split in [("harm", "harmful"), ("safe", "benign")]:
        for row in ds[split]:
            rows.append(
                {
                    "id": f"jbb_{split}_{int(row['Index']):04d}",
                    "label": label,
                    "source": "jailbreakbench",
                    "prompt": row["Goal"],
                    "meta": {
                        "split": split,
                        "index": int(row["Index"]),
                        "behavior": row.get("Behavior"),
                        "category": row.get("Category"),
                        "source": row.get("Source"),
                    },
                }
            )
    return rows


def build_xstest() -> list[dict]:
    from datasets import load_from_disk

    ds = load_from_disk(str(ROOT / "datasets" / "raw" / "xstest"))
    rows = []
    for row in ds["prompts"]:
        prompt_type = row["type"]
        is_harm = prompt_type.startswith("contrast_")
        rows.append(
            {
                "id": f"xstest_{row['id']}",
                "label": "harm" if is_harm else "safe",
                "source": "xstest_prompts",
                "prompt": row["prompt"],
                "meta": {
                    "xstest_id": row["id"],
                    "type": prompt_type,
                    "label_rule": "contrast_* => harm; all other XSTest prompt types => safe",
                },
            }
        )
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_advbench_alpaca(seed: int, max_pairs: int | None) -> list[dict]:
    from datasets import load_from_disk

    adv = read_csv_rows(ROOT / "datasets" / "raw" / "advbench" / "harmful_behaviors.csv")
    alpaca = list(load_from_disk(str(ROOT / "datasets" / "raw" / "alpaca"))["train"])
    n = len(adv) if max_pairs is None else min(len(adv), max_pairs)
    rng = random.Random(seed)
    safe_rows = alpaca[:]
    rng.shuffle(safe_rows)
    safe_rows = safe_rows[:n]

    rows = []
    for idx, row in enumerate(adv[:n]):
        rows.append(
            {
                "id": f"advbench_harm_{idx:04d}",
                "label": "harm",
                "source": "advbench",
                "prompt": row["goal"],
                "meta": {"row_index": idx},
            }
        )
    for idx, row in enumerate(safe_rows):
        rows.append(
            {
                "id": f"alpaca_safe_for_advbench_{idx:04d}",
                "label": "safe",
                "source": "alpaca",
                "prompt": alpaca_prompt(row),
                "meta": {"selection": "seeded_shuffle", "seed": seed, "safe_index": idx},
            }
        )
    return rows


def build_harmbench_alpaca(seed: int, max_pairs: int | None) -> list[dict]:
    from datasets import load_from_disk

    harmbench = read_csv_rows(ROOT / "datasets" / "raw" / "harmbench" / "harmbench_behaviors_text_all.csv")
    alpaca = list(load_from_disk(str(ROOT / "datasets" / "raw" / "alpaca"))["train"])
    n = len(harmbench) if max_pairs is None else min(len(harmbench), max_pairs)
    rng = random.Random(seed + 1)
    safe_rows = alpaca[:]
    rng.shuffle(safe_rows)
    safe_rows = safe_rows[:n]

    rows = []
    for idx, row in enumerate(harmbench[:n]):
        behavior = row["Behavior"].strip()
        context = row.get("ContextString", "").strip()
        prompt = f"{context}\n\n{behavior}" if context else behavior
        rows.append(
            {
                "id": f"harmbench_harm_{idx:04d}",
                "label": "harm",
                "source": "harmbench",
                "prompt": prompt,
                "meta": {
                    "row_index": idx,
                    "behavior_id": row.get("BehaviorID"),
                    "functional_category": row.get("FunctionalCategory"),
                    "semantic_category": row.get("SemanticCategory"),
                    "tags": row.get("Tags"),
                    "has_context": bool(context),
                },
            }
        )
    for idx, row in enumerate(safe_rows):
        rows.append(
            {
                "id": f"alpaca_safe_for_harmbench_{idx:04d}",
                "label": "safe",
                "source": "alpaca",
                "prompt": alpaca_prompt(row),
                "meta": {"selection": "seeded_shuffle", "seed": seed + 1, "safe_index": idx},
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    counts = {}
    sources = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
        sources[row["source"]] = sources.get(row["source"], 0) + 1
    return {"n_records": len(rows), "labels": counts, "sources": sources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="datasets/processed/ood_clean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-broad-pairs", type=int, default=None)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    datasets = {
        "ood_jailbreakbench_clean": build_jailbreakbench(),
        "ood_xstest_clean": build_xstest(),
        "ood_advbench_alpaca_clean": build_advbench_alpaca(args.seed, args.max_broad_pairs),
        "ood_harmbench_alpaca_clean": build_harmbench_alpaca(args.seed, args.max_broad_pairs),
    }

    manifest = {
        "seed": args.seed,
        "max_broad_pairs": args.max_broad_pairs,
        "note": "Clean OOD datasets materialized from raw sources only; legacy processed OOD artifacts are intentionally excluded.",
        "datasets": {},
    }
    for name, rows in datasets.items():
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        manifest["datasets"][name] = {"path": str(path.relative_to(ROOT)), **summarize(rows)}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
