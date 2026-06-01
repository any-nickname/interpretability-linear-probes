"""
Train a layer-wise D1 probe on the active manual In-the-Wild train/eval split.

This is intentionally separate from src/train_probes.py, which targets the old
D1 protocol with heldout/topic/surface/distribution/XSTest eval sets.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def labels_to_y(labels) -> np.ndarray:
    return np.array([1 if label == "harm" else 0 for label in labels], dtype=int)


def slice_layer(acts: torch.Tensor, layer: int) -> np.ndarray:
    return acts[:, layer, :].to(torch.float32).numpy()


def evaluate(probe, scaler, x_eval: np.ndarray, y_eval: np.ndarray) -> dict:
    xs_eval = scaler.transform(x_eval)
    proba = probe.predict_proba(xs_eval)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_eval, proba)),
        "accuracy": float(accuracy_score(y_eval, pred)),
        "f1": float(f1_score(y_eval, pred, zero_division=0)),
        "precision": float(precision_score(y_eval, pred, zero_division=0)),
        "recall": float(recall_score(y_eval, pred, zero_division=0)),
        "mean_p_harm": float(proba.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-name", default="itw_manual_d1_train")
    parser.add_argument("--eval-name", default="itw_manual_d1_eval")
    parser.add_argument("--act-dir", default="data/activations")
    parser.add_argument("--out-dir", default="data/probes/itw_manual_d1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    act_dir = ROOT / args.act_dir / args.model
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = act_dir / f"{args.train_name}.pt"
    eval_path = act_dir / f"{args.eval_name}.pt"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    train = torch.load(train_path, weights_only=False)
    eval_data = torch.load(eval_path, weights_only=False)

    y_train = labels_to_y(train["labels"])
    y_eval = labels_to_y(eval_data["labels"])
    n_layers = int(train["n_layers"])
    d_model = int(train["d_model"])

    rows = []
    layer_payload = []
    for layer in range(n_layers):
        x_train = slice_layer(train["activations"], layer)
        x_eval = slice_layer(eval_data["activations"], layer)

        scaler = StandardScaler()
        xs_train = scaler.fit_transform(x_train)
        probe = LogisticRegression(
            random_state=args.seed,
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
        )
        probe.fit(xs_train, y_train)

        metrics = evaluate(probe, scaler, x_eval, y_eval)
        row = {
            "model": args.model,
            "layer": layer,
            "n_train": int(len(y_train)),
            "n_eval": int(len(y_eval)),
            "train_pos": int(y_train.sum()),
            "eval_pos": int(y_eval.sum()),
            **metrics,
        }
        rows.append(row)
        layer_payload.append(
            {
                "layer": layer,
                "coef": torch.tensor(probe.coef_[0], dtype=torch.float32),
                "intercept": float(probe.intercept_[0]),
                "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
                "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
                "metrics": metrics,
            }
        )
        print(
            f"layer {layer:>2}: AUROC={metrics['auroc']:.4f} "
            f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}",
            flush=True,
        )

    csv_path = out_dir / f"{args.model}_layers.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["auroc"])
    summary = {
        "model": args.model,
        "train_name": args.train_name,
        "eval_name": args.eval_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "best_layer": int(best["layer"]),
        "best_auroc": float(best["auroc"]),
        "best_accuracy": float(best["accuracy"]),
        "best_f1": float(best["f1"]),
    }
    json_path = out_dir / f"{args.model}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pt_path = out_dir / f"{args.model}.pt"
    torch.save(
        {
            "model": args.model,
            "train_name": args.train_name,
            "eval_name": args.eval_name,
            "n_layers": n_layers,
            "d_model": d_model,
            "layers": layer_payload,
            "summary": summary,
        },
        pt_path,
    )

    print(f"best layer={best['layer']} AUROC={best['auroc']:.4f}")
    print(f"wrote {csv_path.relative_to(ROOT)}")
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {pt_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
