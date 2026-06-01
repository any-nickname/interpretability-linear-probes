"""
Materialize the active manual In-the-Wild pairs into D1 train/eval JSONL files.

Input pair rows contain only:
    {"harm_id": "itw_0000", "safe_prompt": "..."}

This script resolves the harmful prompt from
datasets/processed/in_the_wild_source_prompts.jsonl and writes prompt-level
records suitable for activation extraction:
    id, pair_id, label, source, prompt, meta
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("C:/Users/Vladi/Desktop/projs/interpetability")
SOURCE_PATH = ROOT / "datasets" / "processed" / "in_the_wild_source_prompts.jsonl"
PAIR_DIR = ROOT / "datasets" / "pairs"
OUT_DIR = ROOT / "datasets" / "processed"


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def collect_pair_files() -> list[Path]:
    files = []
    for path in sorted(PAIR_DIR.glob("in_the_wild_manual_safe_pairs_*.jsonl")):
        if "user_0000_0009" in path.name:
            continue
        files.append(path)
    return files


def materialize_pair(harm_row: dict, safe_prompt: str, split: str, pair_file: str):
    pair_id = harm_row["id"]
    common_meta = {
        "pair_id": pair_id,
        "split": split,
        "pair_file": pair_file,
        "source_label": harm_row.get("label"),
        "source_meta": harm_row.get("meta", {}),
    }
    harm = {
        "id": f"{pair_id}_harm",
        "pair_id": pair_id,
        "label": "harm",
        "source": "in_the_wild_manual",
        "prompt": harm_row["prompt"],
        "meta": {**common_meta, "side": "harm"},
    }
    safe = {
        "id": f"{pair_id}_safe",
        "pair_id": pair_id,
        "label": "safe",
        "source": "in_the_wild_manual",
        "prompt": safe_prompt,
        "meta": {**common_meta, "side": "safe"},
    }
    return harm, safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--prefix", default="itw_manual_d1")
    args = parser.parse_args()

    sources = {row["id"]: row for row in read_jsonl(SOURCE_PATH)}
    pair_files = collect_pair_files()

    pairs = []
    seen = set()
    for path in pair_files:
        for row in read_jsonl(path):
            harm_id = row["harm_id"]
            if harm_id in seen:
                raise ValueError(f"duplicate harm_id across pair files: {harm_id}")
            if harm_id not in sources:
                raise ValueError(f"missing source prompt for {harm_id}")
            seen.add(harm_id)
            pairs.append(
                {
                    "harm_id": harm_id,
                    "safe_prompt": row["safe_prompt"],
                    "pair_file": path.name,
                }
            )

    pairs = sorted(pairs, key=lambda p: stable_key(p["harm_id"], args.seed))
    train_n = int(round(len(pairs) * args.train_frac))
    train_ids = {p["harm_id"] for p in pairs[:train_n]}

    all_rows = []
    train_rows = []
    eval_rows = []
    manifest_pairs = []

    for pair in sorted(pairs, key=lambda p: p["harm_id"]):
        split = "train" if pair["harm_id"] in train_ids else "eval"
        harm, safe = materialize_pair(
            sources[pair["harm_id"]],
            pair["safe_prompt"],
            split,
            pair["pair_file"],
        )
        all_rows.extend([harm, safe])
        manifest_pairs.append(
            {
                "pair_id": pair["harm_id"],
                "split": split,
                "pair_file": pair["pair_file"],
            }
        )
        if split == "train":
            train_rows.extend([harm, safe])
        else:
            eval_rows.extend([harm, safe])

    all_path = OUT_DIR / f"{args.prefix}_all.jsonl"
    train_path = OUT_DIR / f"{args.prefix}_train.jsonl"
    eval_path = OUT_DIR / f"{args.prefix}_eval.jsonl"
    manifest_path = OUT_DIR / f"{args.prefix}_manifest.json"

    write_jsonl(all_path, all_rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train_frac": args.train_frac,
                "n_pairs": len(pairs),
                "n_train_pairs": len(train_rows) // 2,
                "n_eval_pairs": len(eval_rows) // 2,
                "pair_files": [str(path.relative_to(ROOT)) for path in pair_files],
                "pairs": manifest_pairs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"pairs={len(pairs)} train_pairs={len(train_rows)//2} eval_pairs={len(eval_rows)//2}")
    print(f"wrote {all_path.relative_to(ROOT)}")
    print(f"wrote {train_path.relative_to(ROOT)}")
    print(f"wrote {eval_path.relative_to(ROOT)}")
    print(f"wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
