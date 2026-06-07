"""
Generate Qwen responses for the manual ITW D2-core stage.

This script only collects model responses. It does not assign D2 labels.
Response-behavior labels must be assigned manually in a separate review step.
"""

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path("C:/Users/Vladi/Desktop/projs/interpetability")
DEPS = ROOT / ".deps" / "python"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer


MODELS = {
    "qwen": "Qwen/Qwen3-1.7B",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def output_path(out_dir: Path, model_key: str, dataset_path: Path) -> Path:
    return out_dir / model_key / f"{dataset_path.stem}_responses.jsonl"


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ids.add(json.loads(line)["id"])
    return ids


def tokenize_prompt(tokenizer, prompt: str, device: str = "cuda") -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    try:
        token_ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    return token_ids.to(device)


def generate_response(model, prompt: str, max_new_tokens: int) -> str:
    tokens = tokenize_prompt(model.tokenizer, prompt)
    prompt_len = tokens.shape[1]
    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            stop_at_eos=True,
            verbose=False,
        )
    generated = output[0, prompt_len:]
    return model.tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "datasets/processed/itw_manual_d1_train.jsonl",
            "datasets/processed/itw_manual_d1_eval.jsonl",
        ],
    )
    parser.add_argument("--out-dir", default="data/responses/itw_manual_d2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_paths = []
    for dataset in args.datasets:
        path = Path(dataset)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        dataset_paths.append(path)

    out_dir = ROOT / args.out_dir
    (out_dir / args.model).mkdir(parents=True, exist_ok=True)

    tasks = []
    for path in dataset_paths:
        rows = read_jsonl(path)
        if args.limit is not None:
            rows = rows[: args.limit]
        out_path = output_path(out_dir, args.model, path)
        seen = load_existing_ids(out_path)
        todo = [row for row in rows if row["id"] not in seen]
        tasks.append((path, out_path, todo, len(rows), len(seen)))

    total_todo = sum(len(todo) for _, _, todo, _, _ in tasks)
    print(f"Total remaining responses: {total_todo}", flush=True)
    for path, out_path, todo, n_rows, n_seen in tasks:
        print(
            f"- {path.name}: rows={n_rows}, already_done={n_seen}, remaining={len(todo)}, out={out_path}",
            flush=True,
        )
    if total_todo == 0:
        print("Nothing to do.", flush=True)
        return

    model_name = MODELS[args.model]
    print(f"Loading {model_name}...", flush=True)
    t0 = time.time()
    model = HookedTransformer.from_pretrained(model_name, dtype=torch.float16, device="cuda")
    model.eval()
    print(
        f"Loaded in {time.time() - t0:.1f}s. "
        f"VRAM={torch.cuda.memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )

    t0 = time.time()
    n_done = 0
    for dataset_path, out_path, todo, _, _ in tasks:
        if not todo:
            continue
        with out_path.open("a", encoding="utf-8", newline="\n") as f:
            for row in tqdm(todo, desc=dataset_path.stem, ncols=80):
                try:
                    response = generate_response(model, row["prompt"], args.max_new_tokens)
                    error = None
                except Exception as exc:
                    response = ""
                    error = f"{type(exc).__name__}: {exc}"
                record = {
                    "id": row["id"],
                    "pair_id": row.get("pair_id"),
                    "prompt_label": row.get("label"),
                    "source": row.get("source"),
                    "split": row.get("meta", {}).get("split"),
                    "prompt": row["prompt"],
                    "response": response,
                    "response_error": error,
                    "model_key": args.model,
                    "model": model_name,
                    "max_new_tokens": args.max_new_tokens,
                    "decoding": {"do_sample": False, "temperature": 0.0},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                if n_done % 50 == 0:
                    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(
        f"Done. Generated {n_done} responses in {elapsed:.1f}s "
        f"({n_done / max(elapsed, 1):.2f} responses/sec).",
        flush=True,
    )


if __name__ == "__main__":
    main()
