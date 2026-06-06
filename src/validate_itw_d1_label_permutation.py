"""
Label-permutation sanity check for the manual In-the-Wild D1 activation probe.

The check trains the same layer-wise linear probe after shuffling train labels.
Evaluation is done against the real eval labels. If the pipeline is not leaking
label information, AUROC should be near chance across layers and permutations.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path("C:/Users/Vladi/Desktop/projs/interpetability")
DEPS = ROOT / ".deps" / "python"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def labels_to_y(labels) -> np.ndarray:
    return np.array([1 if label == "harm" else 0 for label in labels], dtype=int)


def slice_layer(acts: torch.Tensor, layer: int) -> np.ndarray:
    return acts[:, layer, :].to(torch.float32).numpy()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--train-name", default="itw_manual_d1_train")
    parser.add_argument("--eval-name", default="itw_manual_d1_eval")
    parser.add_argument("--act-dir", default="data/activations")
    parser.add_argument("--out-dir", default="data/validation/itw_manual_d1/qwen/label_permutation")
    parser.add_argument("--n-permutations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    act_dir = ROOT / args.act_dir / args.model
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train = torch.load(act_dir / f"{args.train_name}.pt", weights_only=False)
    eval_data = torch.load(act_dir / f"{args.eval_name}.pt", weights_only=False)

    y_train_real = labels_to_y(train["labels"])
    y_eval = labels_to_y(eval_data["labels"])
    n_layers = int(train["n_layers"])

    rng = np.random.default_rng(args.seed)
    rows = []
    for perm_idx in range(args.n_permutations):
        y_train = rng.permutation(y_train_real)
        for layer in range(n_layers):
            x_train = slice_layer(train["activations"], layer)
            x_eval = slice_layer(eval_data["activations"], layer)

            scaler = StandardScaler()
            xs_train = scaler.fit_transform(x_train)
            probe = LogisticRegression(
                random_state=args.seed + perm_idx,
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
            )
            probe.fit(xs_train, y_train)
            proba = probe.predict_proba(scaler.transform(x_eval))[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append(
                {
                    "permutation": perm_idx,
                    "layer": layer,
                    "auroc": float(roc_auc_score(y_eval, proba)),
                    "accuracy": float(accuracy_score(y_eval, pred)),
                    "f1": float(f1_score(y_eval, pred, zero_division=0)),
                    "mean_p_harm": float(proba.mean()),
                }
            )
        print(f"permutation {perm_idx + 1}/{args.n_permutations}", flush=True)

    write_csv(
        out_dir / "permutation_layer_metrics.csv",
        rows,
        ["permutation", "layer", "auroc", "accuracy", "f1", "mean_p_harm"],
    )

    by_layer = []
    for layer in range(n_layers):
        vals = [row["auroc"] for row in rows if row["layer"] == layer]
        by_layer.append(
            {
                "layer": layer,
                "mean_auroc": float(np.mean(vals)),
                "std_auroc": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min_auroc": float(np.min(vals)),
                "max_auroc": float(np.max(vals)),
            }
        )
    write_csv(out_dir / "permutation_layer_summary.csv", by_layer, list(by_layer[0].keys()))

    all_aurocs = [row["auroc"] for row in rows]
    summary = {
        "model": args.model,
        "train_name": args.train_name,
        "eval_name": args.eval_name,
        "n_permutations": args.n_permutations,
        "n_layers": n_layers,
        "train_records": int(len(y_train_real)),
        "eval_records": int(len(y_eval)),
        "train_pos_real": int(y_train_real.sum()),
        "eval_pos": int(y_eval.sum()),
        "overall_mean_auroc": float(np.mean(all_aurocs)),
        "overall_std_auroc": float(np.std(all_aurocs, ddof=1)),
        "overall_min_auroc": float(np.min(all_aurocs)),
        "overall_max_auroc": float(np.max(all_aurocs)),
        "best_mean_layer": max(by_layer, key=lambda row: row["mean_auroc"]),
        "worst_mean_layer": min(by_layer, key=lambda row: row["mean_auroc"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Label Permutation Sanity Check",
        "",
        f"- Model: `{args.model}`",
        f"- Permutations: `{args.n_permutations}`",
        f"- Overall mean AUROC: `{summary['overall_mean_auroc']:.4f}`",
        f"- Overall std AUROC: `{summary['overall_std_auroc']:.4f}`",
        f"- Overall min/max AUROC: `{summary['overall_min_auroc']:.4f}` / `{summary['overall_max_auroc']:.4f}`",
        f"- Best mean layer: `{summary['best_mean_layer']['layer']}` (`{summary['best_mean_layer']['mean_auroc']:.4f}`)",
        "",
        "Expected result: AUROC near chance. A high value here would suggest label leakage or another pipeline issue.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
