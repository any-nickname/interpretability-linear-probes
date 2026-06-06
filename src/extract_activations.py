"""
Extract per-layer residual-stream activations from chat-formatted prompts.

For each (model, dataset) combination, produces a .pt file with:
    - activations: tensor[N, n_layers, d_model] (float16)
    - ids, labels, sources, prompts (aligned with activations)
    - n_layers, d_model, model, dataset (metadata)

By default, the activation is taken from `blocks.{i}.hook_resid_post` at the
final token of the chat-formatted prompt (with `add_generation_prompt=True`).
Pass `--pooling mean` to average each layer's activations over all tokens in the
chat-formatted sequence instead.

Usage:
    python src/extract_activations.py --model gemma --datasets datasets/processed/*.jsonl
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
    "gemma": "gemma-2-2b-it",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen": "Qwen/Qwen3-1.7B",
}


def tokenize_prompt(tokenizer, prompt: str, device: str = "cuda") -> torch.Tensor:
    """Apply the model's chat template and tokenize. Returns [1, seq_len] LongTensor."""
    messages = [{"role": "user", "content": prompt}]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids.to(device)
    elif isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"].to(device)
    return token_ids


def extract_dataset(model, records, device="cuda", pooling="last"):
    """Run forward passes, collect one activation vector per prompt and layer."""
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    acts = torch.empty(len(records), n_layers, d_model, dtype=torch.float16)
    hook_names = {f"blocks.{i}.hook_resid_post" for i in range(n_layers)}

    def filter_fn(name: str) -> bool:
        return name in hook_names

    for idx, rec in enumerate(tqdm(records, desc="prompts", ncols=80)):
        tokens = tokenize_prompt(model.tokenizer, rec["prompt"], device=device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=filter_fn,
                return_type=None,
            )

        for li in range(n_layers):
            act = cache[f"blocks.{li}.hook_resid_post"]
            # act shape: [1, seq_len, d_model]
            if pooling == "last":
                pooled = act[0, -1]
            elif pooling == "mean":
                pooled = act[0].mean(dim=0)
            else:
                raise ValueError(f"unknown pooling mode: {pooling}")
            acts[idx, li] = pooled.to(torch.float16).cpu()

        del cache
        if (idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

    return acts


def load_records(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Processed JSONL files to process",
    )
    parser.add_argument(
        "--out-dir", default="data/activations",
        help="Directory (relative to project root) for .pt outputs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if output file exists",
    )
    parser.add_argument(
        "--pooling",
        choices=["last", "mean"],
        default="last",
        help="How to collapse token activations into one vector per prompt/layer",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plan what to do
    tasks = []
    for ds_str in args.datasets:
        ds_path = Path(ds_str)
        if not ds_path.is_absolute():
            ds_path = ROOT / ds_path
        if not ds_path.exists():
            print(f"[skip] {ds_path}: not found", flush=True)
            continue
        out_path = out_dir / f"{ds_path.stem}.pt"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path.name}: already exists (use --force)", flush=True)
            continue
        tasks.append((ds_path, out_path))

    if not tasks:
        print("Nothing to do.", flush=True)
        return

    model_name = MODELS[args.model]
    print(f"Loading {model_name}...", flush=True)
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name, dtype=torch.float16, device="cuda"
    )
    model.eval()
    print(
        f"  loaded in {time.time() - t0:.1f}s. "
        f"layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, "
        f"VRAM={torch.cuda.memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )

    for ds_path, out_path in tasks:
        print(f"\n=== {ds_path.name} -> {out_path.name} ===", flush=True)
        records = load_records(ds_path)
        print(f"  {len(records)} records", flush=True)

        t0 = time.time()
        acts = extract_dataset(model, records, pooling=args.pooling)
        elapsed = time.time() - t0

        torch.save(
            {
                "model": model_name,
                "dataset": ds_path.name,
                "ids": [r["id"] for r in records],
                "labels": [r.get("label") for r in records],
                "sources": [r.get("source") for r in records],
                "prompts": [r["prompt"] for r in records],
                "n_layers": model.cfg.n_layers,
                "d_model": model.cfg.d_model,
                "pooling": args.pooling,
                "activations": acts,
            },
            out_path,
        )
        mb = out_path.stat().st_size / 1024**2
        print(
            f"  saved {tuple(acts.shape)} float16 -> {out_path.name} "
            f"({mb:.1f} MB, {elapsed:.1f}s, "
            f"{len(records) / elapsed:.1f} prompts/sec)",
            flush=True,
        )


if __name__ == "__main__":
    main()
