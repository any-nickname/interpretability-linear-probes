"""
Evaluate the trained manual In-the-Wild D1 activation probe on clean OOD sets.

This script does not retrain the probe. It loads the saved layer-wise linear
probe, applies it to already extracted OOD activations, and reports per-layer
metrics for each OOD dataset.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def labels_to_y(labels) -> np.ndarray:
    return np.array([1 if label == "harm" else 0 for label in labels], dtype=int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    pred = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "mean_p_harm": float(scores.mean()),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def layer_scores(acts: torch.Tensor, layer_payload: dict, layer: int) -> np.ndarray:
    x = acts[:, layer, :].to(torch.float32).numpy()
    mean = layer_payload["scaler_mean"].numpy()
    scale = layer_payload["scaler_scale"].numpy()
    coef = layer_payload["coef"].numpy()
    intercept = float(layer_payload["intercept"])
    logits = ((x - mean) / scale) @ coef + intercept
    return sigmoid(logits)


def dataset_name_from_path(path: Path) -> str:
    name = path.stem
    return name.removesuffix("_clean")


def evaluate_dataset(path: Path, probe_payload: dict, out_dir: Path, focus_layers: list[int]) -> dict:
    data = torch.load(path, weights_only=False)
    y_true = labels_to_y(data["labels"])
    n_layers = int(data["n_layers"])
    acts = data["activations"]
    dataset = dataset_name_from_path(path)

    layer_rows = []
    prediction_rows = []
    focus_scores = {}
    for layer in range(n_layers):
        scores = layer_scores(acts, probe_payload["layers"][layer], layer)
        row = {
            "dataset": dataset,
            "layer": layer,
            "n_records": int(len(y_true)),
            "n_harm": int(y_true.sum()),
            "n_safe": int(len(y_true) - y_true.sum()),
            **metrics_from_scores(y_true, scores),
        }
        layer_rows.append(row)
        if layer in focus_layers:
            focus_scores[layer] = scores

    best = max(layer_rows, key=lambda row: row["auroc"])
    dataset_dir = out_dir / dataset
    write_csv(
        dataset_dir / "layer_metrics.csv",
        layer_rows,
        [
            "dataset",
            "layer",
            "n_records",
            "n_harm",
            "n_safe",
            "auroc",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "mean_p_harm",
        ],
    )

    for idx, rec_id in enumerate(data["ids"]):
        row = {
            "id": rec_id,
            "label": data["labels"][idx],
            "y": int(y_true[idx]),
            "prompt_preview": " ".join(data["prompts"][idx].split())[:360],
        }
        for layer in focus_layers:
            if layer in focus_scores:
                score = float(focus_scores[layer][idx])
                row[f"layer{layer}_p_harm"] = score
                row[f"layer{layer}_pred"] = "harm" if score >= 0.5 else "safe"
        prediction_rows.append(row)

    prediction_fields = ["id", "label", "y"]
    for layer in focus_layers:
        if layer in focus_scores:
            prediction_fields.extend([f"layer{layer}_p_harm", f"layer{layer}_pred"])
    prediction_fields.append("prompt_preview")
    write_csv(dataset_dir / "focus_layer_predictions.csv", prediction_rows, prediction_fields)

    summary = {
        "dataset": dataset,
        "activation_path": str(path.relative_to(ROOT)),
        "n_records": int(len(y_true)),
        "n_harm": int(y_true.sum()),
        "n_safe": int(len(y_true) - y_true.sum()),
        "best_layer": int(best["layer"]),
        "best_auroc": float(best["auroc"]),
        "best_accuracy": float(best["accuracy"]),
        "best_f1": float(best["f1"]),
        "focus_layers": {str(layer): layer_rows[layer] for layer in focus_layers if layer < len(layer_rows)},
    }
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="data/probes/itw_manual_d1/qwen.pt")
    parser.add_argument("--activation-dir", default="data/activations/ood_clean/qwen")
    parser.add_argument("--out-dir", default="data/validation/itw_manual_d1/qwen/ood_clean/probe_eval")
    parser.add_argument("--focus-layers", default="0,12")
    args = parser.parse_args()

    probe_payload = torch.load(ROOT / args.probe, weights_only=False)
    activation_dir = ROOT / args.activation_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_layers = [int(part) for part in args.focus_layers.split(",") if part.strip()]

    paths = sorted(activation_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No activation files found in {activation_dir}")

    summaries = []
    for path in paths:
        summary = evaluate_dataset(path, probe_payload, out_dir, focus_layers)
        summaries.append(summary)
        print(
            f"{summary['dataset']}: best layer {summary['best_layer']} "
            f"AUROC={summary['best_auroc']:.4f}",
            flush=True,
        )

    write_csv(
        out_dir / "ood_probe_summary.csv",
        [
            {
                "dataset": s["dataset"],
                "n_records": s["n_records"],
                "n_harm": s["n_harm"],
                "n_safe": s["n_safe"],
                "best_layer": s["best_layer"],
                "best_auroc": s["best_auroc"],
                "best_accuracy": s["best_accuracy"],
                "best_f1": s["best_f1"],
                "layer0_auroc": s["focus_layers"].get("0", {}).get("auroc"),
                "layer12_auroc": s["focus_layers"].get("12", {}).get("auroc"),
            }
            for s in summaries
        ],
        [
            "dataset",
            "n_records",
            "n_harm",
            "n_safe",
            "best_layer",
            "best_auroc",
            "best_accuracy",
            "best_f1",
            "layer0_auroc",
            "layer12_auroc",
        ],
    )

    payload = {
        "probe": args.probe,
        "activation_dir": args.activation_dir,
        "out_dir": args.out_dir,
        "focus_layers": focus_layers,
        "datasets": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [
        "# Clean OOD Probe Evaluation",
        "",
        f"- Probe: `{args.probe}`",
        f"- Activations: `{args.activation_dir}`",
        "",
        "| Dataset | n | harm | safe | best layer | best AUROC | layer 0 AUROC | layer 12 AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        layer0 = s["focus_layers"].get("0", {}).get("auroc")
        layer12 = s["focus_layers"].get("12", {}).get("auroc")
        report.append(
            f"| `{s['dataset']}` | {s['n_records']} | {s['n_harm']} | {s['n_safe']} | "
            f"{s['best_layer']} | {s['best_auroc']:.4f} | "
            f"{layer0:.4f} | {layer12:.4f} |"
        )
    report.append("")
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
