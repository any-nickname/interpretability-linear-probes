"""
Evaluate the ITW-trained Qwen D2 probe on the manually labeled small OOD-test.

The probe is not retrained here. All layer-wise classifiers are loaded from the
ITW D2 artifact and applied to clean OOD activations.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("C:/Users/Vladi/Desktop/projs/interpetability")
DEPS = ROOT / ".deps" / "python"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80, 80)
    return 1.0 / (1.0 + np.exp(-values))


def metric_block(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "refusal": int(y.sum()),
        "compliance": int(len(y) - y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "mean_p_refusal": float(proba.mean()),
    }
    if len(set(y.tolist())) >= 2:
        out["auroc"] = float(roc_auc_score(y, proba))
    else:
        out["auroc"] = math.nan
    return out


def load_activation_index(paths: list[Path]) -> tuple[dict[str, torch.Tensor], dict]:
    index = {}
    meta = {}
    for path in paths:
        data = torch.load(path, weights_only=False)
        if not meta:
            meta = {
                "n_layers": int(data["n_layers"]),
                "d_model": int(data["d_model"]),
                "pooling": data.get("pooling"),
            }
        for idx, rec_id in enumerate(data["ids"]):
            if rec_id in index:
                raise ValueError(f"duplicate activation id across OOD files: {rec_id}")
            index[rec_id] = data["activations"][idx]
    return index, meta


def evaluate_layer(layer_payload: dict, x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
    mean = layer_payload["scaler_mean"].to(torch.float32).numpy()
    scale = layer_payload["scaler_scale"].to(torch.float32).numpy()
    coef = layer_payload["coef"].to(torch.float32).numpy()
    intercept = float(layer_payload["intercept"])
    xs = (x - mean) / scale
    proba = sigmoid(xs @ coef + intercept)
    return metric_block(y, proba), proba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="data/probes/itw_manual_d2/qwen.pt")
    parser.add_argument(
        "--labels",
        default="data/responses/d2_ood_small/qwen/qwen_d2_small_ood_test_labeled.jsonl",
    )
    parser.add_argument(
        "--activation-files",
        nargs="+",
        default=[
            "data/activations/ood_clean/qwen/ood_jailbreakbench_clean.pt",
            "data/activations/ood_clean/qwen/ood_xstest_clean.pt",
        ],
    )
    parser.add_argument("--primary-layer", type=int, default=20)
    parser.add_argument("--out-dir", default="data/validation/itw_manual_d2/qwen/ood_small/probe_eval")
    args = parser.parse_args()

    probe_path = ROOT / args.probe
    label_path = ROOT / args.labels
    activation_paths = [ROOT / rel for rel in args.activation_files]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_artifact = torch.load(probe_path, weights_only=False)
    labels = read_jsonl(label_path)
    labels = [row for row in labels if row["behavior_label"] in ("refusal", "compliance")]
    y = np.array([1 if row["behavior_label"] == "refusal" else 0 for row in labels], dtype=int)

    activation_index, activation_meta = load_activation_index(activation_paths)
    missing = [row["id"] for row in labels if row["id"] not in activation_index]
    if missing:
        raise ValueError(f"missing OOD activations for ids: {missing}")

    acts = torch.stack([activation_index[row["id"]] for row in labels]).to(torch.float32)
    n_layers = int(probe_artifact["n_layers"])
    if acts.shape[1] != n_layers:
        raise ValueError(f"activation layer mismatch: acts={acts.shape[1]}, probe={n_layers}")

    layer_rows = []
    all_layer_proba = {}
    for layer_payload in probe_artifact["layers"]:
        layer = int(layer_payload["layer"])
        x = acts[:, layer, :].numpy()
        metrics, proba = evaluate_layer(layer_payload, x, y)
        all_layer_proba[layer] = proba
        layer_rows.append(
            {
                "layer": layer,
                **metrics,
            }
        )

    layers_csv = out_dir / "qwen_d2_small_ood_layers.csv"
    write_csv(layers_csv, layer_rows, list(layer_rows[0].keys()))

    best = max(layer_rows, key=lambda row: row["auroc"])
    primary = next(row for row in layer_rows if row["layer"] == args.primary_layer)

    prediction_rows = []
    for idx, row in enumerate(labels):
        rec = {
            "id": row["id"],
            "source": row.get("source"),
            "prompt_label": row.get("prompt_label"),
            "behavior_label": row["behavior_label"],
            "y_refusal": int(y[idx]),
            f"layer{args.primary_layer}_p_refusal": float(all_layer_proba[args.primary_layer][idx]),
            f"layer{int(best['layer'])}_p_refusal": float(all_layer_proba[int(best["layer"])][idx]),
        }
        prediction_rows.append(rec)
    predictions_csv = out_dir / "qwen_d2_small_ood_predictions.csv"
    write_csv(predictions_csv, prediction_rows, list(prediction_rows[0].keys()))

    group_rows = []
    for group_key in ("source", "prompt_label"):
        for group_value in sorted({row.get(group_key) for row in labels}):
            idxs = [idx for idx, row in enumerate(labels) if row.get(group_key) == group_value]
            yy = y[idxs]
            proba = all_layer_proba[args.primary_layer][idxs]
            group_rows.append(
                {
                    "group_key": group_key,
                    "group_value": group_value,
                    "layer": args.primary_layer,
                    **metric_block(yy, proba),
                }
            )
    groups_csv = out_dir / "qwen_d2_small_ood_primary_layer_groups.csv"
    write_csv(groups_csv, group_rows, list(group_rows[0].keys()))

    summary = {
        "probe": str(probe_path.relative_to(ROOT)),
        "labels": str(label_path.relative_to(ROOT)),
        "activation_files": [str(path.relative_to(ROOT)) for path in activation_paths],
        "n_records_binary": int(len(labels)),
        "behavior_counts": dict(Counter(row["behavior_label"] for row in labels)),
        "prompt_label_behavior_counts": {
            label: dict(
                Counter(row["behavior_label"] for row in labels if row.get("prompt_label") == label)
            )
            for label in sorted({row.get("prompt_label") for row in labels})
        },
        "source_behavior_counts": {
            source: dict(Counter(row["behavior_label"] for row in labels if row.get("source") == source))
            for source in sorted({row.get("source") for row in labels})
        },
        "primary_layer": args.primary_layer,
        "primary_layer_metrics": primary,
        "best_ood_layer_exploratory": int(best["layer"]),
        "best_ood_layer_metrics_exploratory": best,
        "activation_meta": activation_meta,
        "note": (
            "Primary layer is fixed from ITW D2 eval selection. The best OOD layer is "
            "reported as exploratory and should not be treated as a selected model."
        ),
    }
    summary_path = out_dir / "qwen_d2_small_ood_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Qwen D2 Small OOD-Test",
        "",
        "- Probe: ITW-trained D2 response-behavior probe",
        f"- Primary layer: `{args.primary_layer}`",
        "- Positive class: `refusal`",
        "- Negative class: `compliance`",
        f"- Binary OOD records: `{len(labels)}` (`{int(y.sum())}` refusal / `{int(len(y) - y.sum())}` compliance)",
        f"- Primary layer AUROC/accuracy/F1: `{primary['auroc']:.4f}` / `{primary['accuracy']:.4f}` / `{primary['f1']:.4f}`",
        f"- Exploratory best OOD layer: `{int(best['layer'])}` with AUROC `{best['auroc']:.4f}`",
        "",
        "The exploratory best layer is reported only as a diagnostic. It is not a valid layer-selection result for this small test set.",
        "",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {layers_csv.relative_to(ROOT)}")
    print(f"wrote {predictions_csv.relative_to(ROOT)}")
    print(f"wrote {groups_csv.relative_to(ROOT)}")
    print(f"wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
